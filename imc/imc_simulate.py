"""
imc/imc_simulate.py
===================
Stage 1 — Data generation for the quadruple-tank MIMO benchmark.

Generates the open-loop dataset (commanded input -> measured output) used to
train the BiLipREN forward/inverse models. The plant is the four-tank system
(Table 1 parameters) integrated with RK4 and a configurable input delay.

Defaults reproduce the dataset used by the closed-loop figure:
    data/quadruple_tank.npz

Run from anywhere (edit the configuration constants below to change settings):
    python imc/imc_simulate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CLEAN_DIR = Path(__file__).resolve().parent
if str(CLEAN_DIR) not in sys.path:
    sys.path.insert(0, str(CLEAN_DIR))
DATA_DIR = CLEAN_DIR / "data"

from imc_sim_core import quadruple_tank_dynamics, rk4_step  # type: ignore[import]

# ---------------------------------------------------------------------------
# Physical parameters (Table 1) — shared across all datasets
# ---------------------------------------------------------------------------
G = 9.81
A1, A2, A3, A4 = 1.31e-4, 1.51e-4, 9.27e-5, 8.82e-5
S = 0.06
H_MIN = np.array([0.0, 0.0, 0.0, 0.0])
H_MAX = np.array([1.36, 1.36, 1.30, 1.30])
QA_MIN, QA_MAX = 0.0, 9.0e-4
QB_MIN, QB_MAX = 0.0, 1.3e-3


# ---------------------------------------------------------------------------
# Excitation input generator (PRBS with a slow random amplitude envelope)
# ---------------------------------------------------------------------------
PRBS_DWELL_MIN, PRBS_DWELL_MAX = 2, 12     # steps a fast on/off bit is held
ENV_DWELL_MIN, ENV_DWELL_MAX = 20, 80      # steps a slow amplitude level is held
ENV_LOW, ENV_HIGH = 0.35, 1.0              # amplitude range (fraction of pump span)


def _piecewise_random(rng, steps, dwell_min, dwell_max, sampler):
    """Piecewise-constant signal of length `steps`: hold a value drawn from
    `sampler` for a random run of [dwell_min, dwell_max] steps, then resample."""
    values = []
    while len(values) < steps:
        run = int(rng.integers(dwell_min, dwell_max + 1))
        values.extend([sampler()] * run)
    return np.asarray(values[:steps], dtype=np.float64)


def generate_command_sequence(rng, steps):
    """Two-channel excitation: a fast random on/off bit (PRBS) scaled by a slow
    random amplitude envelope, mapped onto each pump range [Q*_MIN, Q*_MAX]."""
    bit_a = _piecewise_random(rng, steps, PRBS_DWELL_MIN, PRBS_DWELL_MAX, lambda: rng.integers(0, 2))
    bit_b = _piecewise_random(rng, steps, PRBS_DWELL_MIN, PRBS_DWELL_MAX, lambda: rng.integers(0, 2))
    env_a = _piecewise_random(rng, steps, ENV_DWELL_MIN, ENV_DWELL_MAX, lambda: rng.uniform(ENV_LOW, ENV_HIGH))
    env_b = _piecewise_random(rng, steps, ENV_DWELL_MIN, ENV_DWELL_MAX, lambda: rng.uniform(ENV_LOW, ENV_HIGH))
    qa = QA_MIN + env_a * bit_a * (QA_MAX - QA_MIN)
    qb = QB_MIN + env_b * bit_b * (QB_MAX - QB_MIN)
    return np.stack([qa, qb], axis=1)


# ---------------------------------------------------------------------------
# Dataset configuration (defaults reproduce the shipped dataset)
# ---------------------------------------------------------------------------
GAMMA_A, GAMMA_B = 0.5, 0.6   # gamma_a, gamma_b — four-tank valve split ratios
DELAY_STEPS = 5               # input delay tau = 125 s (5 samples at dt = 25 s)
NUM_TRAJ = 1000               # number of trajectories
NUM_STEPS = 400               # trajectory length (discrete time steps)
DT = 25.0                     # plant sampling period (s); RK4 integration step for the continuous tank ODE
SEED = 0


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DATA_DIR / "quadruple_tank.npz"

    plant = dict(
        g=G, S=S, a1=A1, a2=A2, a3=A3, a4=A4,
        gamma_a=GAMMA_A, gamma_b=GAMMA_B,
        h_min=H_MIN, h_max=H_MAX,
    )
    rng = np.random.default_rng(SEED)

    U_seq = np.zeros((NUM_TRAJ, NUM_STEPS, 2))
    U_applied_seq = np.zeros((NUM_TRAJ, NUM_STEPS, 2))
    Y_seq = np.zeros((NUM_TRAJ, NUM_STEPS, 2))
    X_seq = np.zeros((NUM_TRAJ, NUM_STEPS, 4))

    for i in range(NUM_TRAJ):
        x = np.clip(rng.uniform(0.2, 1.0, size=4), H_MIN, H_MAX)
        u_cmd_seq = generate_command_sequence(rng, NUM_STEPS)
        u_buffer = [np.array([QA_MIN, QB_MIN]) for _ in range(DELAY_STEPS)]
        for k in range(NUM_STEPS):
            u = u_cmd_seq[k]
            if DELAY_STEPS > 0:
                u_buffer.append(u.copy())
                u_applied = u_buffer.pop(0)
            else:
                u_applied = u
            U_seq[i, k] = u
            U_applied_seq[i, k] = u_applied
            Y_seq[i, k] = x[:2]
            X_seq[i, k] = x
            if k < NUM_STEPS - 1:
                x = rk4_step(
                    lambda xx, uu: quadruple_tank_dynamics(xx, uu, plant),
                    x, u_applied, DT,
                )
                x = np.clip(x, H_MIN, H_MAX)

    np.savez(
        save_path,
        U=U_seq,
        U_applied=U_applied_seq,
        Y=Y_seq,
        X=X_seq,
        dt=DT,
        params=dict(
            g=G, S=S,
            a1=A1, a2=A2, a3=A3, a4=A4,
            gamma_a=GAMMA_A, gamma_b=GAMMA_B,
            h_min=H_MIN, h_max=H_MAX,
            qa_min=QA_MIN, qa_max=QA_MAX,
            qb_min=QB_MIN, qb_max=QB_MAX,
            input_mode="prbs_envelope",
            prbs_dwell_min=PRBS_DWELL_MIN, prbs_dwell_max=PRBS_DWELL_MAX,
            envelope_dwell_min=ENV_DWELL_MIN, envelope_dwell_max=ENV_DWELL_MAX,
            envelope_low=ENV_LOW, envelope_high=ENV_HIGH,
            input_delay_steps=DELAY_STEPS,
        ),
    )
    print(f"Dataset saved: {save_path}")
    print("U:", U_seq.shape, " U_applied:", U_applied_seq.shape,
          " Y:", Y_seq.shape, " X:", X_seq.shape)


if __name__ == "__main__":
    main()
