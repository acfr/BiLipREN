"""
flow/flow_training.py
=====================
Normalizing-flow trainer for the BiLipREN composition (CompREN) on the
trajectory dataset.  Maximum-likelihood (NLL) objective:

    NLL = sum_t [ 0.5||z_t||^2  -  log|det J_t| ]

where z_t = G(x_t) is the per-step latent and J_t the layer Jacobian product.
DATA-ONLY: trains and saves the best model + normalization stats; no plotting.
Inputs are state_action features (trajectories ++ actions, nu=4).

Run from the project root:
    python flow/flow_training.py
"""
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

_HERE    = Path(__file__).resolve().parent
_ROOT    = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from BiLipRENs.ren_composition import CompREN

# ── Paths ───────────────────────────────────────────────────────────────────────────────
_MODELS  = _HERE / "models"
_DATA    = _HERE / "data"
for _d in (_MODELS, _DATA):
    _d.mkdir(parents=True, exist_ok=True)
DATASET = _ROOT / "surrogate_cost" / "data" / "dataset.npz"

# ── Architecture / training hyperparameters ──────────────────────────────────
NX, NV, NUM_LAYERS = 10, 32, 12
LOWER, UPPER       = 0.01, 20.0
EPOCHS     = 500
BATCH      = 128
LR         = 1e-3
WD         = 1e-4
SEED       = 0


def build_states(batch_size, nx, num_layers):
    state = jnp.zeros((batch_size, nx)) + 0.1
    return [[state] for _ in range(num_layers)]


def iterate_minibatches(data, batch_size, rng, shuffle):
    idx = np.arange(len(data))
    if shuffle:
        rng.shuffle(idx)
    usable = len(idx) // batch_size * batch_size
    idx = idx[:usable]
    for i in range(0, usable, batch_size):
        yield data[idx[i:i + batch_size]]


def iterate_minibatches_full(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def make_cosine_schedule(lr, total_steps):
    warmup = max(1, min(300, total_steps // 10, total_steps - 1))
    return optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.05, peak_value=lr, warmup_steps=warmup,
        decay_steps=max(1, total_steps), end_value=lr * 0.05,
    )


# ── Load dataset (state_action: trajectories ++ actions) ─────────────────────
_ds = np.load(str(DATASET))
trajectories = np.asarray(_ds["trajectories"], dtype=np.float32)   # (N, T, 2)
actions      = np.asarray(_ds["actions"],      dtype=np.float32)   # (N, T, 2)
inputs_all   = np.concatenate([trajectories, actions], axis=-1)    # (N, T, 4)
nu = int(inputs_all.shape[-1])

n_total = inputs_all.shape[0]
n_train = int(0.8 * n_total)

rng_split = np.random.default_rng(SEED)
perm = rng_split.permutation(n_total)
inputs_shuffled = inputs_all[perm]
train_data = inputs_shuffled[:n_train]
val_data   = inputs_shuffled[n_train:]

# Per-feature z-score normalization fitted on training data only.
feat_mean = train_data.mean(axis=(0, 1)).astype(np.float32)
feat_std  = np.maximum(train_data.std(axis=(0, 1)), 1e-6).astype(np.float32)
train_data_norm = ((train_data - feat_mean) / feat_std).astype(np.float32)
val_data_norm   = ((val_data   - feat_mean) / feat_std).astype(np.float32)

# Persist the exact split + normalization stats so flow_plot.py reproduces it.
np.savez_compressed(
    _DATA / "flow_train_subset.npz",
    train_data=train_data, val_data=val_data,
    feat_mean=feat_mean, feat_std=feat_std,
)
print(f"[Data] {DATASET.name}  N={n_total}  train={n_train}  val={n_total - n_train}  "
      f"nu={nu} nx={NX} nv={NV} L={NUM_LAYERS}  bounds=({LOWER},{UPPER})")

# ── Build model ──────────────────────────────────────────────────────────────
key_init = jax.random.PRNGKey(SEED)
model = CompREN(nu, NX, NV, NUM_LAYERS, LOWER, UPPER, dyn_orth=False)

init_inputs = jnp.asarray(train_data_norm[:BATCH, 0, :])
init_states = build_states(BATCH, NX, NUM_LAYERS)
params = model.init(key_init, init_states, init_inputs)

steps_per_epoch = max(1, len(train_data_norm) // BATCH)
schedule  = make_cosine_schedule(LR, max(1, EPOCHS * steps_per_epoch))
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=WD),
)
opt_state = optimizer.init(params)


@jax.jit
def nll_batch(p, batch):
    b = batch.shape[0]
    states = build_states(b, NX, NUM_LAYERS)
    inputs_t = jnp.transpose(batch, (1, 0, 2))

    def step(carry, x_t):
        s, pp = carry
        ns, z_t, jacobians = model.apply(pp, s, x_t, return_jacobians=True)
        det_j = jnp.stack([jnp.linalg.det(j["jacobian"]) for j in jacobians])
        neg_log_pz = jnp.sum(0.5 * jnp.square(z_t), axis=1)
        log_det = jnp.sum(jnp.log(jnp.abs(det_j) + 1e-12), axis=0)
        return (ns, pp), (neg_log_pz, log_det)

    (_, _), (neg_log_pz_t, log_det_t) = jax.lax.scan(step, (states, p), inputs_t)
    objective = -neg_log_pz_t + log_det_t
    return -jnp.mean(jnp.sum(objective, axis=0))


@jax.jit
def train_step(p, o, batch):
    loss, grads = jax.value_and_grad(nll_batch)(p, batch)
    updates, o = optimizer.update(grads, o, p)
    p = optax.apply_updates(p, updates)
    return p, o, loss


# ── Training loop ────────────────────────────────────────────────────────────
best_val, best_epoch, no_improve = float("inf"), -1, 0
last_val = float("nan")
rng_np = np.random.default_rng(SEED)

for epoch in range(EPOCHS):
    batch_losses = []
    for batch in iterate_minibatches(train_data_norm, BATCH, rng_np, shuffle=True):
        params, opt_state, loss = train_step(params, opt_state, jnp.asarray(batch))
        batch_losses.append(loss)
    train_epoch = float(jnp.mean(jnp.stack(batch_losses))) if batch_losses else float("nan")

    if (epoch % 5 == 0) or (epoch == EPOCHS - 1):
        val_losses = [float(nll_batch(params, jnp.asarray(b)))
                      for b in iterate_minibatches_full(val_data_norm, BATCH)]
        if val_losses:
            last_val = float(np.mean(val_losses))
            if (best_val - last_val) > 1e-4:
                best_val, best_epoch, no_improve = last_val, epoch, 0
                with open(_MODELS / "flow_best.pkl", "wb") as f:
                    pickle.dump(params, f)
            else:
                no_improve += 1
            if no_improve >= 20:
                print(f"[Early stop] epoch {epoch}, best val={best_val:.6f} (epoch {best_epoch})")
                break

    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:4d}  train={train_epoch:.4f}  val={last_val:.4f}  "
              f"best_val={best_val:.4f}  no_improve={no_improve}")

if best_epoch < 0:
    with open(_MODELS / "flow_best.pkl", "wb") as f:
        pickle.dump(params, f)

print(f"[Done] best val={best_val:.6f} at epoch {best_epoch}  →  {_MODELS / 'flow_best.pkl'}")
