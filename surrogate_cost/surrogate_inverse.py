import numpy as np
import jax.numpy as jnp
import os

import jax
import pickle

from BiLipRENs.ren_composition_inverse import CompRENinv
from BiLipRENs.utils import create_states

# ── Architecture: must match training ────────────────────────────────────────
num_layers = 5
nu, nx, nv = 2, 8, 64
lower, upper = 0.1, 48    # must match the trained BiLipREN checkpoint

# ── Dataset (for horizon length) ─────────────────────────────────────────────
from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
_DATA = _HERE / "data"
_RESULTS = _HERE / "results"
_RESULTS.mkdir(parents=True, exist_ok=True)
training_data = np.load(str(_DATA / 'dataset.npz'))
time_step = training_data['actions'].shape[1]   # full horizon T

# ── Load trained params ───────────────────────────────────────────────────────
params_path = str(_DATA / "BiLipREN_best.pkl")
with open(params_path, 'rb') as f:
    _ckpt = pickle.load(f)
params = _ckpt['params'] if 'params' in _ckpt and 'c_raw' in _ckpt.get('params', {}) else _ckpt

# ── G⁻¹(0) inverse: feed zeros into the inverse REN to recover argmin of F ────
n_traj = 1
states_init = create_states(
    batch_size=n_traj,
    state_size=nx,
    num_layers=num_layers,
    dyn_orth=False,
    dyn_mult=1,
)
model_inv = CompRENinv(nu, nx, nv, num_layers, lower, upper, dyn_orth=False)
params_inv_r = CompRENinv.reverse_params(params, num_layers, dyn_orth=False)
params_for_inv = {'params': params_inv_r['params'], 'c_raw': params['c_raw']}

@jax.jit
def scan_fn_inv(carry, inputs):
    states, p = carry
    new_states, new_outputs = model_inv.apply({'params': p['params']}, states, inputs)
    return (new_states, p), new_outputs

_zero_in = jnp.zeros((time_step, n_traj, nu), dtype=jnp.float32)
_, ren_out_inv = jax.lax.scan(scan_fn_inv, (states_init, params_for_inv), _zero_in)
actions_inv = ren_out_inv[:, 0, :]                   # (T, nu)

# ── Save inverse actions (G^-1(0)) for the plotting script ───────────────────
_bilipren_path = str(_DATA / 'bilipren.pkl')
with open(_bilipren_path, 'wb') as f:
    pickle.dump({'actions_inv': np.asarray(actions_inv)}, f)
print(f'Saved inverse actions (G^-1(0)) -> {_bilipren_path}')
