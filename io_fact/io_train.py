"""
io_fact/io_train.py
===================
Train the inner-outer ORTHREN model (dynamic orthogonal block placed at the
output) used to produce results/io_fact.pdf.

Reads  : data/inner_outer_data.pkl
Writes : models/inner_outer_params.pkl   (weights + meta)
         data/inner_outer_loss.pkl        (test-loss history)
"""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from BiLipRENs.ren_composition import ORTHREN
from BiLipRENs.utils import data_generator, l2_norm_loss

# --- Core network parameters (paper notation) ------------------------------
NUM_LAYERS = 1     # number of REN layers
NX = 10            # REN internal states
NV = 32            # REN neurons
MU = 0.1           # strong I/O monotonicity bound (paper: mu)
NU = 8             # Lipschitz upper bound          (paper: nu)
DYN_MULT = 10      # dynamic-orthogonal state multiplier

# --- Training hyper-parameters ---------------------------------------------
EPOCHS = 150
BATCH = 64
SAMPLES = 2000     # trajectories used for training (rest are held out for test)
TIME_STEP = 50
LR = 1e-3
GRAD_CLIP = 10.0
SEED = 0

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "inner_outer_data.pkl", "rb") as f:
    inputs_data, outputs_data = pickle.load(f)

n_in = int(inputs_data.shape[-1])

key = jax.random.key(SEED)
data_key, init_key = jax.random.split(key)

# Train / test split: test = the BATCH trajectories right after the training set.
train_in = inputs_data[:SAMPLES, :TIME_STEP, :]
train_out = outputs_data[:SAMPLES, :TIME_STEP, :]
test_in = jnp.transpose(inputs_data[SAMPLES:SAMPLES + BATCH, :TIME_STEP, :], (1, 0, 2))
test_out = jnp.transpose(outputs_data[SAMPLES:SAMPLES + BATCH, :TIME_STEP, :], (1, 0, 2))


def init_states(batch_size):
    # DynOrth sits at the output: only the last layer carries the dyn state.
    ren = jnp.zeros((batch_size, NX))
    dyn = jnp.zeros((batch_size, DYN_MULT * NX))
    states = [[ren] for _ in range(NUM_LAYERS - 1)]
    states.append([ren, dyn])
    return states


model = ORTHREN(n_in, NX, NV, NUM_LAYERS, MU, NU,
                dyn_orth=False, dyn_orth_at_output=True,
                dyn_orth_state_multiplier=DYN_MULT)
params = model.init(init_key, init_states(BATCH), train_in[:BATCH, 0, :])

# Learning-rate schedule sized to the real number of update steps.
steps = max(1, EPOCHS * ((SAMPLES + BATCH - 1) // BATCH))
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=LR,
    warmup_steps=max(1, steps // 20), decay_steps=steps, end_value=LR * 1e-2)
solver = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(schedule))
opt_state = solver.init(params)


@jax.jit
def loss_fn(p, states, batch):
    u, y = batch
    def step(carry, x):
        st, pp = carry
        new_st, out = model.apply(pp, st, x)
        return (new_st, pp), out
    _, pred = jax.lax.scan(step, (states, p), u)
    return l2_norm_loss(y, pred)


@jax.jit
def update(p, opt_state, states, batch):
    grads = jax.grad(loss_fn)(p, states, batch)
    updates, opt_state = solver.update(grads, opt_state, p)
    return optax.apply_updates(p, updates), opt_state


loss_history = []
for epoch, _, batch_in, batch_out in data_generator(train_in, train_out, BATCH, EPOCHS, data_key):
    u = jnp.transpose(batch_in, (1, 0, 2))
    y = jnp.transpose(batch_out, (1, 0, 2))
    params, opt_state = update(params, opt_state, init_states(batch_in.shape[0]), (u, y))
    test_loss = float(loss_fn(params, init_states(BATCH), (test_in, test_out)))
    loss_history.append(test_loss)
    print(f"Epoch {epoch}  test loss {test_loss:.4e}")

payload = dict(params)
payload["meta"] = {
    "num_layers": NUM_LAYERS, "n_in": n_in, "nx": NX, "nv": NV,
    "mu": MU, "nu": NU, "dyn_mult": DYN_MULT,
    "epochs": EPOCHS, "batch": BATCH, "samples": SAMPLES,
    "time_step": TIME_STEP, "lr": LR, "seed": SEED,
}
with open(MODEL_DIR / "inner_outer_params.pkl", "wb") as f:
    pickle.dump(payload, f)
with open(DATA_DIR / "inner_outer_loss.pkl", "wb") as f:
    pickle.dump(loss_history, f)
print("Saved model and loss history.")
