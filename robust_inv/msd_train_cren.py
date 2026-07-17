"""
robust_inv/msd_train_cren.py
============================
Train a forward Contracting REN on the MSD dataset
(`data/msd_data{,_val}.pkl`):

  forward : u → y   (matches CompREN forward direction)

Model size (nx=8, nv=180) is chosen so total params (~39k) match the
BiLipREN CompREN(nu=1, nx=16, nv=64, num_layers=4) baseline.

Usage:
    python robust_inv/msd_train_cren.py
"""

import os
import pickle
import argparse

from BiLipRENs.utils import add_device_cli_arg, configure_device_from_args

_cli = argparse.ArgumentParser()
add_device_cli_arg(_cli)
_cli_args = _cli.parse_args()
configure_device_from_args(_cli_args)

import jax
import jax.numpy as jnp
import optax
from robustnn import ren_jax as ren

jax.config.update("jax_default_matmul_precision", "highest")

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(__file__)
DATA_TRAIN = os.path.join(_HERE, "data", "msd_data.pkl")
DATA_VAL   = os.path.join(_HERE, "data", "msd_data_val.pkl")
OUT_DIR    = os.path.join(_HERE, "models", "cren")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
# Matched to lip_sweep settings on the MSD dataset.
# Model size chosen so total params (~39.0k) ≈ CompREN(nx=16, nv=64, 4 layers)
# baseline (38,882 params); the C-REN with nx=8, nv=180 has 39,049 params.
NX        = 8
NV        = 180
BATCHES   = 10                # 100 samples / 10 = 10 batches/epoch
EPOCHS    = 200
SEED      = 0

# LR schedule  (warmup-cosine, identical shape to lip_sweep)
LR_PEAK   = 3e-4
LR_END    = 1e-5
LR_WARMUP = 100

# Gradient clipping (global L2 norm)
GRAD_CLIP = 1.0

# ── Load data ────────────────────────────────────────────────────────────────
with open(DATA_TRAIN, "rb") as f:
    train_u, train_y = pickle.load(f)        # (N, T, 1) each
with open(DATA_VAL, "rb") as f:
    val_u, val_y = pickle.load(f)            # (M, T, 1) each

N_train, T, _ = train_u.shape
n_val         = val_u.shape[0]
print(f"Train: {N_train} × {T}   Val: {n_val} × {T}")


def _count_params(d):
    total = 0
    for v in d.values():
        if isinstance(v, jnp.ndarray):
            total += v.size
        elif isinstance(v, dict):
            total += _count_params(v)
    return total


def train_cren():
    """Forward C-REN: u → y."""
    tr_in, tr_out = train_u, train_y
    va_in, va_out = val_u,   val_y
    tag = "fwd"

    print(f"\n{'='*60}\nContractingREN  [FORWARD]\n{'='*60}")

    rng = jax.random.key(SEED)
    rng, key_init, key_data = jax.random.split(rng, 3)

    model = ren.ContractingREN(
        input_size=1, state_size=NX, features=NV, output_size=1,
    )

    x0_train = jnp.zeros((BATCHES, NX))
    x0_val   = jnp.zeros((n_val,   NX))
    params   = model.init(key_init, x0_train, jnp.array(tr_in[:BATCHES, 0, :]))

    # Re-initialise the D22 feedthrough randomly (library hard-codes zeros_init).
    from flax.core import unfreeze
    params = unfreeze(params)
    rng, key_d22 = jax.random.split(rng)
    params["params"]["D22"] = jax.nn.initializers.lecun_normal()(
        key_d22, params["params"]["D22"].shape, params["params"]["D22"].dtype
    )

    print(f"  NX={NX}  NV={NV}  Params={_count_params(params)}")

    # Optimiser:  warmup_cosine_decay + global-norm clipping
    total_steps = EPOCHS * max(N_train // BATCHES, 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR_PEAK,
        warmup_steps=LR_WARMUP, decay_steps=total_steps, end_value=LR_END,
    )
    solver = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adam(learning_rate=schedule),
    )
    opt_state = solver.init(params)

    # Pre-transpose val to (T, n_val, 1)
    test_u = jnp.transpose(jnp.array(va_in),  axes=(1, 0, 2))
    test_y = jnp.transpose(jnp.array(va_out), axes=(1, 0, 2))

    @jax.jit
    def loss_fn(params, x0, u_T, y_T):
        _, y_pred = model.simulate_sequence(params, x0, u_T)
        diff = y_T - y_pred
        rms = jnp.sqrt(jnp.mean(jnp.sum(diff**2, axis=-1), axis=0))
        ref = jnp.sqrt(jnp.mean(jnp.sum(y_T**2,  axis=-1), axis=0))
        return jnp.mean(rms / (ref + 1e-8))

    @jax.jit
    def update_step(params, opt_state, x0, u_T, y_T):
        loss_val, grads = jax.value_and_grad(loss_fn)(params, x0, u_T, y_T)
        updates, new_opt = solver.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss_val

    train_log, val_log = [], []
    n_batches = N_train // BATCHES
    rng_data  = key_data

    for ep in range(1, EPOCHS + 1):
        rng_data, subkey = jax.random.split(rng_data)
        perm = jax.random.permutation(subkey, N_train)

        ep_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * BATCHES : (b + 1) * BATCHES]
            u_b = jnp.transpose(jnp.array(tr_in[idx]),  axes=(1, 0, 2))
            y_b = jnp.transpose(jnp.array(tr_out[idx]), axes=(1, 0, 2))
            params, opt_state, batch_loss = update_step(
                params, opt_state, x0_train, u_b, y_b)
            ep_loss += float(batch_loss)
        ep_loss /= n_batches

        val_loss = float(loss_fn(params, x0_val, test_u, test_y))
        train_log.append(ep_loss)
        val_log.append(val_loss)
        print(f"  Epoch {ep:3d}/{EPOCHS}  Train: {ep_loss:.6f}  Val: {val_loss:.6f}")

    params_path = os.path.join(OUT_DIR, f"msd_cren_{tag}_params.pkl")
    loss_path   = os.path.join(OUT_DIR, f"msd_cren_{tag}_loss.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    with open(loss_path, "wb") as f:
        pickle.dump({"train": train_log, "val": val_log,
                     "epochs": EPOCHS, "batches_per_epoch": n_batches}, f)
    print(f"  Saved → {params_path}")


if __name__ == "__main__":
    train_cren()
    print("\nAll done.")
