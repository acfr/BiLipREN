# ============================================================================
# Cost-surrogate trainer for two non-injective recurrent cells, selected on the CLI:
#     cren -> Contracting REN (robustnn.ren_jax.ContractingREN), features=166
#     lstm -> standard LSTM (flax.linen.LSTMCell + Dense readout), hidden H=91
#   Neither cell is a bijection, so G^-1(0) is computed NUMERICALLY as
#   argmin_u 0.5||G(u)||^2.
#
# Run from project root:
#     python surrogate_cost/surrogate_training_ablation.py [cren|lstm]
# ============================================================================
import os
import glob
import importlib
import sys
import numpy as np
import jax.numpy as jnp

import jax
import optax
import flax.linen as fnn
import pickle

# ── model selector (CLI) ─────────────────────────────────────────────────────
MODEL = (sys.argv[1].lower() if len(sys.argv) > 1 else "cren")
if MODEL not in ("cren", "lstm"):
    print(f"ERROR: unknown model '{MODEL}'. Use 'cren' or 'lstm'.")
    sys.exit(1)
print(f"[ablation] MODEL = {MODEL}")

if importlib.util.find_spec("robustnn") is None:
    print("ERROR: 'robustnn' not found. Install:\n"
          "  pip install git+https://github.com/acfr/RobustNeuralNetworks.git")
    sys.exit(1)
from robustnn import ren_jax as ren

from BiLipRENs.utils import data_generator

from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
_DATA = _HERE / "data"
_RESULTS = _HERE / "results"
_RESULTS.mkdir(parents=True, exist_ok=True)
_DATA_NPZ = str(_DATA / 'dataset.npz')
training_data = np.load(_DATA_NPZ)

# Per-sample fit weight w_i = exp(-(cost_i - cost_min)/_FIT_TAU): down-weights far,
#   high-cost trajectories so the fit concentrates on the near-minimum cost bowl.
_FIT_TAU = 100.0

action_data = training_data['actions']
cost_data    = training_data['costs']
_CMIN        = float(cost_data.min())   # cost minimum (reweighting baseline)

print(f"[Data] source = {_DATA_NPZ}  N={len(cost_data)}  T={action_data.shape[1]}  "
      f"cost_min={cost_data.min():.1f} (idx {int(np.argmin(cost_data))})")

rng = jax.random.key(0)
rng, key1, key2, key3, key4, key5 = jax.random.split(rng, 6)

num_layers = 5

nu, nx, nv = 2, 8, 64
# CREN features sized to a fixed ~34k-parameter budget.
NV_CREN = 166
# LSTM hidden size sized to ~match the same budget; carry = concat(cell, hidden) = 2*H.
H_LSTM    = 91
# unified surrogate-state dimension (CREN: plain nx; LSTM: concat(c,h)=2H)
STATE_DIM = nx if MODEL == "cren" else 2 * H_LSTM

epoch = 200
batches = 256
samples = len(cost_data)   # use all trajectories
beta_w = 0.0               # 0 = plain MSE (uniform)
time_step = action_data.shape[1]

seed = 0
np.random.seed(seed)

idx = np.arange(len(cost_data))
np.random.shuffle(idx)

train_idx = idx[:samples]   # all samples for training (no held-out split)

training_in  = action_data[train_idx, :, :]
training_out = cost_data[train_idx].astype(np.float64)   # raw cost (no log transform)
eval_every   = samples // batches   # batches per epoch

# ── Near-minimum eval subset: bottom 256 by cost (fits states_zero_init) ─────
_eval_idx  = np.argsort(training_out)[:batches]
eval_in    = jnp.transpose(training_in[_eval_idx], axes=(1, 0, 2))   # (T, 256, nu)
eval_out   = training_out[_eval_idx]

_DATA_MIN_IDX  = int(np.argmin(cost_data))
_DATA_MIN_COST = float(cost_data[_DATA_MIN_IDX])
# c is a freely-learnable offset, initialised at the dataset-min cost.
_c_init = float(_DATA_MIN_COST)


def get_c(params):
    return jax.nn.softplus(params['c_raw'])   # c = softplus(c_raw) > 0 always

# Standard LSTM cost-surrogate cell.  Same call signature as the CREN:
#   G(state, u) -> (new_state, output) so the existing jax.lax.scan over a
#   trajectory and the entire downstream pipeline are UNCHANGED.  The recurrent
#   carry is the concatenated [cell, hidden] vector (dim 2H); a Dense layer maps
#   the hidden state to the nu-dim residual whose 0.5||·||^2 IS the PLNet cost
#   head q.  Like the CREN it is NOT a bijection (no analytic inverse).
class LSTMSurrogate(fnn.Module):
    hidden_size: int
    output_size: int
    @fnn.compact
    def __call__(self, state, inputs):
        H = self.hidden_size
        c, h = state[:, :H], state[:, H:]
        new_carry, y = fnn.LSTMCell(features=H)((c, h), inputs)
        out = fnn.Dense(self.output_size)(y)
        return jnp.concatenate(new_carry, axis=-1), out

# Create the surrogate cell selected by MODEL.  States are plain (batch, STATE_DIM)
#   zero arrays (CREN: nx; LSTM: concat(c,h)=2H) — no layered create_states tuple.
inputs_init = training_in[0:batches, 0]
states_zero_init = jnp.zeros((batches, STATE_DIM), dtype=jnp.float32)
if MODEL == "cren":
    model = ren.ContractingREN(
        input_size  = nu,
        state_size  = nx,
        features    = NV_CREN,
        output_size = nu,
    )
else:
    model = LSTMSurrogate(
        hidden_size = H_LSTM,
        output_size = nu,
    )
params_ren = model.init(key2, states_zero_init, inputs_init)

# stable inverse-softplus for LARGE raw c (expm1 overflows >88 in fp32):
#   softplus^{-1}(x) = x + log(-expm1(-x)) = x + log1p(-exp(-x)).  For x~1863 this = x.
_cf = jnp.asarray(_c_init, dtype=jnp.float32)
_c_raw_init = _cf + jnp.log(-jnp.expm1(-_cf))   # softplus(c_raw_init) ≈ _c_init
param_c = {'c_raw': _c_raw_init}  # trainable via softplus; c = softplus(c_raw) > 0 guaranteed
params = {**params_ren, **param_c}
schedule_fn = optax.warmup_exponential_decay_schedule(
    init_value=0.0,        # starting lr
    peak_value=3e-3,       # Phase-1 fit needs aggressive lr at high conditioning
    warmup_steps=500,      # warmup for 500 steps
    transition_steps=2000, # decay half-life; LR reaches end_value after ~180k steps (~epoch 5000)
    decay_rate=0.95,       # exponential rate
    end_value=1e-5,        # stop decaying after reaching this
)
# c is TRAINABLE: use the same adam schedule for c_raw.
param_labels = {'params': 'ren', 'c_raw': 'c'}
# c is LEARNABLE in both models; the c-drift penalty in `loss` keeps it near c_target.
_c_transform = optax.adam(learning_rate=5e-2)
solver = optax.multi_transform(
    transforms={
        'ren': optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule_fn),
        ),
        'c':   _c_transform,          # c LEARNABLE (softplus); constrained by the c-drift penalty in `loss`.
    },
    param_labels=param_labels,
)
opt_state = solver.init(params)

@jax.jit
def scan_fn(carry, inputs):
    states, params = carry
    # params contains extra key 'c_raw'; pass only the model variables
    new_states, new_outputs = model.apply({'params': params['params']}, states, inputs)
    return (new_states, params), new_outputs


@jax.jit
def loss(params, states_init, batch_data):
    training_in, training_out = batch_data
    _, ren_out = jax.lax.scan(scan_fn, (states_init, params), training_in)
    q = 0.5 * jnp.sum(ren_out**2, axis=(0, 2))
    pl_loss = q + get_c(params)
    error = training_out - pl_loss
    # [cost-reweight] per-sample weight w = exp(-(cost - cmin)/tau): down-weights FAR
    #   high-cost trajectories so the cost bowl around the dataset-min stays well-fit.
    w_fit    = jnp.exp(-(training_out - _CMIN) / _FIT_TAU)
    fit_loss = jnp.sum(w_fit * error**2) / (jnp.sum(w_fit) + 1e-8)
    return fit_loss


@jax.jit
def predict_quadratic(params, states_init, training_in):
    _, ren_out = jax.lax.scan(scan_fn, (states_init, params), training_in)
    return 0.5 * jnp.sum(ren_out**2, axis=(0, 2))


@jax.jit
def predict_cost(params, states_init, training_in):
    return predict_quadratic(params, states_init, training_in) + get_c(params)


@jax.jit
def metrics(params, states_init, batch_data):
    training_in, training_out = batch_data
    pred = predict_cost(params, states_init, training_in)
    err = pred - training_out
    abs_err = jnp.abs(err)
    return {
        'mae': jnp.mean(abs_err),
        'rmse': jnp.sqrt(jnp.mean(err**2)),
        'bias': jnp.mean(err),
    }


@jax.jit
def update(params, opt_state, x0, inputs):
    loss_val, grads = jax.value_and_grad(loss, argnums=0)(params, x0, inputs)
    updates, opt_state = solver.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_val

CKPT_EVERY = 50    # checkpoint period (epochs)
PATIENCE = 300     # early stopping: stop if no improvement for this many epochs

# ── Output directory named by experiment parameters ───────────────────────────
_bw   = str(beta_w).replace('.', 'p')
_tau_s = str(_FIT_TAU).replace('.', 'p')
# distinct exp_tag per model so the cren/lstm OUT_DIRs never collide / resume the
#   wrong checkpoints (the carry-state shape differs between the CREN and LSTM).
if MODEL == "cren":
    exp_tag = f"CREN_ABLATION_nl{num_layers}_nu{nu}_nx{nx}_nvCREN{NV_CREN}_ep{epoch}_bs{batches}_fitTAU{_tau_s}"
else:
    exp_tag = f"LSTM_ABLATION_nl{num_layers}_nu{nu}_nx{nx}_H{H_LSTM}_ep{epoch}_bs{batches}_fitTAU{_tau_s}"


OUT_DIR = str(_RESULTS / exp_tag)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[Config] OUT_DIR = {OUT_DIR}")

# ── Resume: find latest numbered checkpoint in OUT_DIR ────────────────────────
start_epoch = 0
best_test_rmse = float('inf')
best_params = params
loss_value = []
no_improve_count = 0

ckpt_files = sorted(glob.glob(os.path.join(OUT_DIR, "ckpt_ep*.pkl")))
if ckpt_files:
    latest_ckpt = ckpt_files[-1]
    with open(latest_ckpt, "rb") as f:
        ckpt = pickle.load(f)
    params           = ckpt['params']
    opt_state        = ckpt['opt_state']
    loss_value       = ckpt['loss_value']
    start_epoch      = ckpt['epoch'] + 1
    best_test_rmse   = ckpt.get('best_test_rmse', float('inf'))
    best_params      = ckpt.get('best_params', params)
    no_improve_count = ckpt.get('no_improve_count', 0)
    print(f"[Resume] {os.path.basename(latest_ckpt)}, epoch={ckpt['epoch']}, best_rmse={best_test_rmse:.4f}, no_improve={no_improve_count}")
else:
    print("[Fresh] No checkpoint found, starting from scratch.")

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch, batch_idx, batch_in, batch_out in data_generator(training_in, training_out, batches, epoch, key1, drop_last=True):
    if epoch < start_epoch:   # skip already-trained epochs
        continue
    in_T = jnp.transpose(batch_in, axes=(1, 0, 2))
    params, opt_state, loss_fn = update(params, opt_state, states_zero_init,
                                         (in_T, jnp.array(batch_out)))
    loss_value.append(loss_fn)
    if batch_idx % eval_every == 0:
        eval_loss = loss(params, states_zero_init, (eval_in, eval_out))
        train_metrics = metrics(params, states_zero_init, (in_T, jnp.array(batch_out)))
        eval_metrics  = metrics(params, states_zero_init, (eval_in, eval_out))
        c_value = get_c(params)
        # select best purely by fit quality (eval RMSE); the c-drift penalty in the
        #   loss already keeps c near the dataset-min cost.
        combined = float(eval_metrics['rmse'])

        is_best = combined < best_test_rmse
        if is_best:
            best_test_rmse = combined
            best_params = params
            no_improve_count = 0
            best_fname = os.path.join(OUT_DIR, "best_params.pkl")
            with open(best_fname, "wb") as f:
                pickle.dump(dict(best_params), f)
            print(f"  → [Best] Saved best_params.pkl  ep={epoch}  eval_rmse={best_test_rmse:.4f}  c={c_value:.4f}")
        else:
            no_improve_count += 1
        print(
            f"Epoch {epoch}, "
            f"train_loss: {loss_fn:.4f}, eval_loss: {eval_loss:.4f}, "
            f"c: {c_value:.4f}, beta_w: {beta_w:.2f}, "
            f"train_rmse: {train_metrics['rmse']:.4f}, eval_rmse: {eval_metrics['rmse']:.4f}, "
            f"train_bias: {train_metrics['bias']:.4f}, eval_bias: {eval_metrics['bias']:.4f}"
            + (" [best]" if is_best else f" [no_improve: {no_improve_count}/{PATIENCE}]")
        )
        if no_improve_count >= PATIENCE:
            print(f"[Early Stop] No improvement for {PATIENCE} epochs. Stopping at epoch {epoch}.")
            break
        # ── Periodic checkpoint: one file per checkpoint, never overwritten ──────
        if epoch % CKPT_EVERY == 0:
            ckpt = {
                'params':           params,
                'opt_state':        opt_state,
                'loss_value':       loss_value,
                'epoch':            epoch,
                'best_test_rmse':   best_test_rmse,
                'best_params':      best_params,
                'no_improve_count': no_improve_count,
            }
            ckpt_fname = os.path.join(OUT_DIR, f"ckpt_ep{epoch:04d}.pkl")
            with open(ckpt_fname, "wb") as f:
                pickle.dump(ckpt, f)
            loss_fname = os.path.join(OUT_DIR, "loss.pkl")
            with open(loss_fname, "wb") as f:
                pickle.dump(loss_value, f)
            print(f"[Checkpoint] Saved {os.path.basename(ckpt_fname)} (no_improve: {no_improve_count}/{PATIENCE})")

# ── Final save (unique name, never overwritten) ──────────────────────────────
final_fname = os.path.join(OUT_DIR, f"final_ep{epoch:04d}_rmse{best_test_rmse:.4f}.pkl")
with open(final_fname, "wb") as f:
    pickle.dump(dict(best_params), f)
loss_fname = os.path.join(OUT_DIR, "loss.pkl")
with open(loss_fname, "wb") as f:
    pickle.dump(loss_value, f)
print(f"[Done] best_rmse={best_test_rmse:.4f} \u2192 {os.path.basename(final_fname)}")
print(f"[Done] All outputs saved to: {OUT_DIR}")
