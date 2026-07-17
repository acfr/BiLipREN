"""
imc/imc_sim_core.py
===================
IMC closed-loop simulator core (library only).

Provides `simulate_imc_closed_loop` and `make_random_step_ref`, shared by
`plot.py` and `imc_sweep_table.py`. This module is not runnable on its own.
"""

import os
import sys

import jax.numpy as jnp
import numpy as np

CLEAN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CLEAN_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from BiLipRENs.ren_composition import CompREN
from BiLipRENs.ren_composition_inverse import CompRENinv
from BiLipRENs.utils import normalize_to_unit


def model_tag(lower, upper):
    """Filename tag of a trained forward model (must match imc_lip_sweep.py)."""
    return f"{lower}-{upper}"


def denormalize_from_unit(z, z_min, z_max, eps=1e-12):
    return 0.5 * (z + 1.0) * (z_max - z_min + eps) + z_min


def quadruple_tank_dynamics(x, u, p):
    x = np.clip(x, p["h_min"], p["h_max"])
    h1, h2, h3, h4 = x
    qa, qb = u

    dh1 = -p["a1"] / p["S"] * np.sqrt(2.0 * p["g"] * h1) + p["a3"] / p["S"] * np.sqrt(2.0 * p["g"] * h3) + p["gamma_a"] / p["S"] * qa
    dh2 = -p["a2"] / p["S"] * np.sqrt(2.0 * p["g"] * h2) + p["a4"] / p["S"] * np.sqrt(2.0 * p["g"] * h4) + p["gamma_b"] / p["S"] * qb
    dh3 = -p["a3"] / p["S"] * np.sqrt(2.0 * p["g"] * h3) + (1.0 - p["gamma_b"]) / p["S"] * qb
    dh4 = -p["a4"] / p["S"] * np.sqrt(2.0 * p["g"] * h4) + (1.0 - p["gamma_a"]) / p["S"] * qa
    return np.array([dh1, dh2, dh3, dh4], dtype=np.float64)


def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_states_fwd(batch_size, nx, num_layers, dyn_mult, dyn_orth, dyn_orth_at_output=False):
    ren_state = jnp.zeros((batch_size, nx), dtype=jnp.float32)
    dyn_state = jnp.zeros((batch_size, dyn_mult * nx), dtype=jnp.float32)
    if dyn_orth:
        states = [[ren_state, dyn_state]]
        for _ in range(1, num_layers):
            states.append([ren_state])
        return states
    if dyn_orth_at_output:
        states = [[ren_state] for _ in range(num_layers - 1)]
        states.append([ren_state, dyn_state])
        return states
    return [[ren_state] for _ in range(num_layers)]


def build_states_inv(batch_size, nx, num_layers):
    ren_state = jnp.zeros((batch_size, nx), dtype=jnp.float32)
    return [[ren_state] for _ in range(num_layers)]


def make_random_step_ref(steps, out_min, out_max, seed, hold_min, hold_max, margin=0.05):
    if hold_min <= 0 or hold_max < hold_min:
        raise ValueError("Invalid step-hold range")
    lo = float(out_min + margin * (out_max - out_min))
    hi = float(out_max - margin * (out_max - out_min))
    if hi <= lo:
        raise ValueError("Invalid reference range after margin")

    rng = np.random.default_rng(seed)
    ref = np.zeros((steps, 2), dtype=np.float64)
    idx = 0
    cur = rng.uniform(lo, hi, size=(2,))
    while idx < steps:
        dur = int(rng.integers(hold_min, hold_max + 1))
        ref[idx : min(steps, idx + dur)] = cur
        idx += dur
        cur = rng.uniform(lo, hi, size=(2,))
    return ref


def aligned_tracking_metrics(y, ref, lag):
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if lag >= len(y):
        raise ValueError("lag must be smaller than sequence length")
    if lag == 0:
        y_cmp = y
        ref_cmp = ref
    else:
        y_cmp = y[lag:]
        ref_cmp = ref[:-lag]

    err = y_cmp - ref_cmp
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    fit_denom = np.linalg.norm(ref_cmp - np.mean(ref_cmp, axis=0, keepdims=True), axis=0) + 1e-12
    fit_num = np.linalg.norm(err, axis=0)
    fit_per_dim = (1.0 - fit_num / fit_denom) * 100.0
    return rmse, mae, fit_per_dim


def simulate_imc_closed_loop(
    params_blob,
    x_init,
    y_ref,
    data_params,
    dt,
    warmup_steps,
    lpf_beta=0.92,
    robust_filter_beta=0.5,
    robust_filter_order=2,
    output_noise_std=0.0,
    output_noise_seed=42,
):
    params = {"params": params_blob["params"]}
    meta = params_blob.get("meta", {})

    num_layers = int(meta["num_layers"])
    nu = int(meta["nu"])
    nx = int(meta["nx"])
    nv = int(meta["nv"])
    lower = float(meta["lower"])
    upper = float(meta["upper"])
    dyn_mult = int(meta.get("dyn_mult", 38))
    dyn_orth = bool(meta.get("dyn_orth", False))
    dyn_orth_at_output = bool(meta.get("dyn_orth_at_output", False))

    out_min = float(np.min(np.array(data_params["h_min"], dtype=np.float64)[:2]))
    out_max = float(np.max(np.array(data_params["h_max"], dtype=np.float64)[:2]))
    u1_min, u1_max = float(data_params["qa_min"]), float(data_params["qa_max"])
    u2_min, u2_max = float(data_params["qb_min"]), float(data_params["qb_max"])
    input_delay_steps = int(data_params.get("input_delay_steps", 0))

    plant_params = {
        "g": float(data_params["g"]),
        "S": float(data_params["S"]),
        "a1": float(data_params["a1"]),
        "a2": float(data_params["a2"]),
        "a3": float(data_params["a3"]),
        "a4": float(data_params["a4"]),
        "gamma_a": float(data_params["gamma_a"]),
        "gamma_b": float(data_params["gamma_b"]),
        "h_min": np.array(data_params["h_min"], dtype=np.float64),
        "h_max": np.array(data_params["h_max"], dtype=np.float64),
    }

    model = CompREN(
        nu,
        nx,
        nv,
        num_layers,
        lower,
        upper,
        dyn_orth=dyn_orth,
        dyn_orth_at_output=dyn_orth_at_output,
        dyn_orth_state_multiplier=dyn_mult,
    )
    inv_model = CompRENinv(nu, nx, nv, num_layers, lower, upper, dyn_orth=dyn_orth)
    inv_params = CompRENinv.reverse_params(params_blob, num_layers, dyn_orth=dyn_orth)

    # Compute H0 (DynOrth DC-gain) for IMC compensation.
    # With delay-free training (U_applied→Y), DynOrth captures only input direction
    # rotation (not time-delay), so H0 is accurate at all operating frequencies.
    if dyn_orth or dyn_orth_at_output:
        from BiLipRENs.utils import cayley as _cayley
        dyn_key = "models_0" if dyn_orth else f"models_{num_layers + 2}"
        _X = jnp.array(params_blob["params"][dyn_key]["X"])
        _G = _cayley(_X)
        _ns = dyn_mult * nx
        _A, _B, _C, _D = _G[:_ns, :_ns], _G[:_ns, _ns:], _G[_ns:, :_ns], _G[_ns:, _ns:]
        _H0 = _C @ jnp.linalg.solve(jnp.eye(_ns, dtype=_A.dtype) - _A, _B) + _D if _ns > 0 else _D
        H0_comp = _H0
    else:
        H0_comp = None

    states_m = build_states_fwd(1, nx, num_layers, dyn_mult, dyn_orth, dyn_orth_at_output)
    states_c = build_states_inv(1, nx, num_layers)

    y_warm_ref = np.array(x_init[:2], dtype=np.float64)
    total_steps = int(y_ref.shape[0]) + int(max(0, warmup_steps))
    d_hat = jnp.zeros((1, nu), dtype=jnp.float32)
    u_norm = jnp.zeros((1, nu), dtype=jnp.float32)
    x = np.array(x_init, dtype=np.float64).copy()

    if not (0.0 <= float(lpf_beta) < 1.0):
        raise ValueError("lpf_beta must be in [0, 1).")
    if not (0.0 <= float(robust_filter_beta) < 1.0):
        raise ValueError("robust_filter_beta must be in [0, 1).")
    if int(robust_filter_order) <= 0:
        raise ValueError("robust_filter_order must be >= 1.")

    u_delay_buffer = [np.array([u1_min, u2_min], dtype=np.float64) for _ in range(input_delay_steps)]
    _noise_rng = np.random.default_rng(output_noise_seed)
    log_ref, log_y, log_y_true, log_ym, log_u, log_d, log_warm = [], [], [], [], [], [], []
    robust_states = [jnp.zeros((1, nu), dtype=jnp.float32) for _ in range(int(robust_filter_order))]
    # Feed applied (delayed) input to the internal model so it mirrors the plant.
    u_applied_norm = jnp.zeros((1, nu), dtype=jnp.float32)

    for k in range(total_steps):
        is_warmup = k < warmup_steps
        ref_idx = max(0, k - warmup_steps)
        y_true = np.array(x[:2], dtype=np.float64)
        y = y_true.copy()
        if output_noise_std > 0.0:
            y = y + _noise_rng.normal(0.0, output_noise_std * (out_max - out_min), size=2)
        r_phys = y_warm_ref if is_warmup else np.array(y_ref[ref_idx], dtype=np.float64)

        y_norm = jnp.array(normalize_to_unit(y, out_min, out_max), dtype=jnp.float32)[None, :]
        r_norm = jnp.array(normalize_to_unit(r_phys, out_min, out_max), dtype=jnp.float32)[None, :]

        # Feed applied (delayed) input to the internal model so it mirrors the plant.
        states_m, y_m_norm = model.apply(params, states_m, u_applied_norm)
        a_y = y_norm - y_m_norm
        d_hat = float(lpf_beta) * d_hat + (1.0 - float(lpf_beta)) * a_y

        c_in = r_norm - d_hat
        if dyn_orth_at_output and H0_comp is not None:
            # gmin-dynorth: rotate target into pre-DynOrth space before inversion
            # y = G_min(u) @ H0^T, so G_min(u) = c_in @ H0 => u = CompRENinv(c_in @ H0)
            c_in_inv = c_in @ H0_comp
        else:
            c_in_inv = c_in
        states_c, u_norm_raw = inv_model.apply(inv_params, states_c, c_in_inv)
        # Apply H0 compensation for dynorth-gmin: u = v_d @ H0 (apply after inv)
        if not dyn_orth_at_output and H0_comp is not None:
            u_norm_raw = u_norm_raw @ H0_comp
        u_norm_raw = jnp.clip(u_norm_raw, -1.0, 1.0)
        u_f = u_norm_raw
        for _i in range(int(robust_filter_order)):
            robust_states[_i] = float(robust_filter_beta) * robust_states[_i] + (1.0 - float(robust_filter_beta)) * u_f
            robust_states[_i] = jnp.clip(robust_states[_i], -1.0, 1.0)
            u_f = robust_states[_i]
        u_norm = jnp.clip(u_f, -1.0, 1.0)

        u_phys = np.array(
            [
                denormalize_from_unit(float(u_norm[0, 0]), u1_min, u1_max),
                denormalize_from_unit(float(u_norm[0, 1]), u2_min, u2_max),
            ],
            dtype=np.float64,
        )
        u_phys = np.clip(u_phys, [u1_min, u2_min], [u1_max, u2_max])

        if input_delay_steps > 0:
            u_delay_buffer.append(u_phys.copy())
            u_applied = u_delay_buffer.pop(0)
        else:
            u_applied = u_phys
        # Update normalized applied input for the next model step.
        u_applied_norm = jnp.array(
            [normalize_to_unit(u_applied[0], u1_min, u1_max),
             normalize_to_unit(u_applied[1], u2_min, u2_max)],
            dtype=jnp.float32,
        )[None, :]

        x = rk4_step(lambda xx, uu: quadruple_tank_dynamics(xx, uu, plant_params), x, u_applied, float(dt))
        x = np.clip(x, plant_params["h_min"], plant_params["h_max"])

        y_m = np.array(
            [
                denormalize_from_unit(float(y_m_norm[0, 0]), out_min, out_max),
                denormalize_from_unit(float(y_m_norm[0, 1]), out_min, out_max),
            ],
            dtype=np.float64,
        )

        log_ref.append(r_phys)
        log_y.append(y)
        log_y_true.append(y_true)
        log_ym.append(y_m)
        log_u.append(u_phys)
        log_d.append(np.array(d_hat[0], dtype=np.float64))
        log_warm.append(is_warmup)

    ref_arr = np.array(log_ref)
    y_arr = np.array(log_y)
    y_true_arr = np.array(log_y_true)
    ym_arr = np.array(log_ym)
    u_arr = np.array(log_u)
    d_arr = np.array(log_d)
    warm_mask = np.array(log_warm, dtype=bool)

    eval_ref = ref_arr[~warm_mask]
    eval_y = y_arr[~warm_mask]
    eval_y_true = y_true_arr[~warm_mask]
    eval_ym = ym_arr[~warm_mask]
    eval_u = u_arr[~warm_mask]
    eval_d = d_arr[~warm_mask]

    lag_eval = input_delay_steps
    rmse_raw, mae_raw, fit_raw = aligned_tracking_metrics(eval_y, eval_ref, lag=0)
    rmse, mae, fit_lag = aligned_tracking_metrics(eval_y, eval_ref, lag=lag_eval)
    out_range = max(float(out_max) - float(out_min), 1e-12)
    nrmse_raw = rmse_raw / out_range
    nrmse = rmse / out_range
    sat1 = float(np.mean((eval_u[:, 0] <= u1_min + 1e-12) | (eval_u[:, 0] >= u1_max - 1e-12)))
    sat2 = float(np.mean((eval_u[:, 1] <= u2_min + 1e-12) | (eval_u[:, 1] >= u2_max - 1e-12)))

    return {
        "dt": float(dt),
        "input_delay_steps": input_delay_steps,
        "out_min": out_min,
        "out_max": out_max,
        "warmup_steps": int(warmup_steps),
        "lag_eval": int(lag_eval),
        "ref": ref_arr,
        "y": y_arr,
        "ym": ym_arr,
        "u": u_arr,
        "d": d_arr,
        "is_warmup": warm_mask,
        "ref_eval": eval_ref,
        "y_eval": eval_y,
        "y_true_eval": eval_y_true,
        "ym_eval": eval_ym,
        "u_eval": eval_u,
        "d_eval": eval_d,
        "rmse_raw": rmse_raw,
        "mae_raw": mae_raw,
        "fit_raw": fit_raw,
        "nrmse_raw": nrmse_raw,
        "rmse": rmse,
        "mae": mae,
        "fit_lag": fit_lag,
        "nrmse": nrmse,
        "sat1": sat1,
        "sat2": sat2,
    }

