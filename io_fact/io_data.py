import numpy as np
import pickle
import jax
from pathlib import Path

# Fixed true-system parameters (defined ONLY here; no other file hard-codes them)
ALPHA = 0.9       # feedback gain in y[k] = ALPHA*tanh(y[k-1]) + u[k-DELAY]
DELAY = 3         # input delay (time steps)
TIME_STEP = 50    # sequence length used for training
NUM_SAMPLES = 2500


def simulate_system(u, y0=0.0):
    y = [y0] * (DELAY + 1)  # Initialize with initial condition repeated for delay
    for k in range(len(u)):
        u_delayed = 0.0 if k < DELAY else u[k - DELAY]
        y_next = ALPHA * np.tanh(y[-1]) + u_delayed
        y.append(y_next)
    return np.array(y[DELAY + 1:])  # Discard the initial repeated values

key = jax.random.PRNGKey(0)
inputs = []
outputs = []

for i in range(NUM_SAMPLES):
    key, subkey = jax.random.split(key)
    u = jax.random.normal(subkey, (TIME_STEP,))
    y = simulate_system(u)
    inputs.append(u)
    outputs.append(y)

inputs = np.array(inputs).reshape(NUM_SAMPLES, TIME_STEP, 1)
outputs = np.array(outputs).reshape(NUM_SAMPLES, TIME_STEP, 1)

training_data = (inputs, outputs)
data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
with open(data_dir / "inner_outer_data.pkl", "wb") as f:
    pickle.dump(training_data, f)
