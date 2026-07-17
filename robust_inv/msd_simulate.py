"""
robust_inv/msd_simulate.py
==========================
Data generation for the n-linked mass-spring-damper (MSD) chain benchmark
(msd_chain class).

The output is packaged as the pickle window format consumed by the lip-sweep /
layers-sweep training scripts in this directory (msd_data.pkl / msd_data_val.pkl).

Pipeline:
  * Integrator:        scipy.integrate.solve_ivp (default RK45, adaptive)
  * Measurement noise: v_sd = 0.05  (Gaussian, added to y only)
  * Process noise:     w_sd = 0.0
  * Input:             random_bit_stream (piecewise-constant)
        u_per = []
        while sum(u_per) < T:
            u_per += [int(period * rand())]
        u_per[-1] -= (sum(u_per) - T)
        u = concat([u_sd*(rand-0.5)*ones((per,1)) for per in u_per])
  * Random IC (train): X0 = 10*(rand(2*N*batches) - 0.5)   (sim_rand_ic)
  * Zero IC   (val)  : X0 = zeros(2*N)                     (simulate)
  * Output:            y = position of last cart + Gaussian(0, v_sd)
  * Sampling:          Ts = 0.5 s  (2 Hz)

Dataset sizes (defaults; override via CLI):
  train  : sim_rand_ic(seq_len=300=150 s, batches=300), u_sd=3.0  → 300 windows
           (shortened from 1000 steps and enlarged 100→300 trajectories so the
            total sample count stays ≈ 90k while tripling IC/input diversity)
  val    : simulate(T=5000), u_sd=3.0, sliced @1000              → 5 windows × 1000
           (kept long so the zero-IC transient is a small fraction of each window)

Output files (data/):
  msd_data.pkl      : (train_in, train_out), each (n_traj, TRAIN_SEQ_LEN, 1)
  msd_data_val.pkl  : (val_in,   val_out),   each (n_win,  VAL_WINDOW_LEN, 1)
"""

import os
import pickle
import argparse
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp


# ============================================================================
# msd_chain  --  n-linked mass-spring-damper chain
# ============================================================================

class msd_chain:
    def __init__(self, N=4, T=5000, Ts=0.5, u_sd=3.0, period=100):
        self.N = N
        self.T = T
        self.Ts = Ts
        self.u_sd = u_sd
        self.period = period

        # System parameters (n-linked MSD chain)
        self.k = np.linspace(1.0, 0.5, N)
        self.c = 0.5 * np.linspace(0.5, 1.0, N)
        self.m = 0.5 * np.linspace(0.5, 1.0, N)

        self.v_sd = 0.05    # measurement noise std
        self.w_sd = 0.0     # process noise std
        self.travel = 2

    # ---- piecewise-constant input -----------------------------------------
    def random_bit_stream(self, T=None):
        if T is None:
            T = self.T
        u_per = []
        while sum(u_per) < T:
            u_per += [int(self.period * np.random.rand())]
        u_per[-1] = u_per[-1] - (sum(u_per) - T)
        u = np.concatenate(
            [self.u_sd * (np.random.rand() - 0.5) * np.ones((per, 1))
             for per in u_per], 0)
        return u  # shape (T, 1)

    # ---- nonlinear spring (piecewise linear, kf=0.25 soft band) ----------
    def spring_func(self, x):
        kf = 0.25
        d = self.travel / 2          # = 1.0
        f = kf * x * (x < d) * (x > -d)
        f = f + (x - d + kf * d) * (x >= d)
        f = f + (x + d - kf * d) * (x <= -d)
        return f

    # ---- continuous-time dynamics ----------------------------------------
    # x is (2*N, batches), u is (1, batches)
    def dynamics(self, x, u, w=0):
        d = x[0::2, :]
        v = x[1::2, :]
        k, c, m = self.k, self.c, self.m

        F = 0 * d

        # First cart
        F[0] = F[0] + (w + u + k[0] * self.spring_func(-d[0])
                       + k[1] * self.spring_func(d[1] - d[0])
                       - c[0] * v[0] + c[1] * (v[1] - v[0]))

        # Middle carts
        F[1:-1] = F[1:-1] + (
            k[1:-1, None] * self.spring_func(d[0:-2, :] - d[1:-1, :]) +
            k[2:,    None] * self.spring_func(d[2:,    :] - d[1:-1, :]) +
            c[1:-1, None] * (v[0:-2, :] - v[1:-1, :]) +
            c[2:,    None] * (v[2:,    :] - v[1:-1, :]))

        # Last cart
        F[-1] = F[-1] + (k[-1, None] * self.spring_func(d[-2, :] - d[-1, :])
                         + c[-1, None] * (v[-2, :] - v[-1, :]))

        dxdt = 0 * x
        dxdt[0::2] = v
        dxdt[1::2] = F / m[:, None]
        return dxdt

    # ---- single trajectory simulation (zero IC) --------------------------
    def simulate(self, u=None):
        if u is None:
            Tend = self.T * self.Ts
            time = np.linspace(0, Tend, self.T)
            u = self.random_bit_stream()
        else:
            T = u.shape[0]
            Tend = self.Ts * T
            time = np.linspace(0, Tend, T)

        u_interp = interp.interp1d(time, u[:, 0])
        x0 = np.zeros((2 * self.N))

        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, u_interp(t)[None])
            return dX.reshape(-1)

        sol = integrate.solve_ivp(dyn, [0.0, Tend], x0, t_eval=time)
        Y = sol['y'][None, -2:-1, :]                     # (1, 1, T)
        Y = Y + np.random.normal(0, self.v_sd, Y.shape)
        u_out = u.T[:, None, :]                          # (1, 1, T)
        return u_out, Y

    # ---- batched simulation with random initial conditions ----------------
    def sim_rand_ic(self, seq_len, batches):
        Tend = seq_len * self.Ts
        time = np.linspace(0, Tend, seq_len)

        u = self.random_bit_stream(batches * seq_len)    # (batches*seq_len, 1)
        u = u.reshape(batches, seq_len)
        u_interp = interp.interp1d(time, u.T, axis=0)

        X0 = 10 * (np.random.rand(2 * self.N * batches) - 0.5)

        def dyn(t, x):
            X = x.reshape(2 * self.N, -1)
            dX = self.dynamics(X, u_interp(t))
            return dX.reshape(-1)

        sol = integrate.solve_ivp(dyn, [0.0, Tend], X0, t_eval=time)
        Y = sol['y'].reshape(2 * self.N, batches, -1).T  # (seq_len, batches, 2N)
        U = u_interp(time)                               # (seq_len, batches)

        U = U.T[:, None, :]                              # (batches, 1, seq_len)
        Y = Y.transpose(1, 2, 0)[:, -2:-1, :]            # (batches, 1, seq_len)
        Y = Y + np.random.normal(0, self.v_sd, Y.shape)
        return U, Y


# ============================================================================
# Windowing  (raw (n_traj,1,T) → sequential (n_win, window_len, 1))
# ============================================================================

# Training windows are now 300 steps (= 150 s at Ts=0.5); validation windows
# stay long (1000 steps) for a stable, low-transient evaluation.
TRAIN_SEQ_LEN  = 300
VAL_WINDOW_LEN = 1000

def sequential_windows(u, y, window_len):
    """
    Slice (n_traj, 1, T) arrays into sequential non-overlapping windows.
    Returns (windows_u, windows_y), each shape (n_traj * floor(T/window_len),
    window_len, 1).
    """
    u = u.transpose(0, 2, 1)   # (n_traj, T, 1)
    y = y.transpose(0, 2, 1)
    n_traj, T, _ = u.shape
    n_win = T // window_len
    u = u[:, : n_win * window_len, :].reshape(n_traj * n_win, window_len, 1)
    y = y[:, : n_win * window_len, :].reshape(n_traj * n_win, window_len, 1)
    return u, y


# ============================================================================
# Main
# ============================================================================

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seq-len", type=int, default=TRAIN_SEQ_LEN,
                        help="Training window length in steps (default 300 = 150 s @ Ts=0.5).")
    parser.add_argument("--train-batches", type=int, default=300,
                        help="Number of training trajectories (default 300; was 100).")
    parser.add_argument("--val-T", type=int, default=5000,
                        help="Validation trajectory length in steps (default 5000).")
    parser.add_argument("--val-window", type=int, default=VAL_WINDOW_LEN,
                        help="Validation window length in steps (default 1000).")
    parser.add_argument("--period", type=int, default=100,
                        help="Piecewise-constant input segment length in steps "
                             "(default 100 = 50 s; lower it for more input "
                             "transitions per short window).")
    parser.add_argument("--train-out", type=str, default="msd_data.pkl",
                        help="Train pickle filename written under data/ "
                             "(default msd_data.pkl; use a custom name to avoid "
                             "clobbering the shared 300-step dataset).")
    parser.add_argument("--skip-val", action="store_true",
                        help="Skip regenerating the validation set (leave the "
                             "shared msd_data_val.pkl untouched).")
    args = parser.parse_args()

    print("=" * 70)
    print("MSD chain generation + window pickling")
    print("=" * 70)

    # ── Training: sim_rand_ic(seq_len, batches), u_sd=3.0 ──────────────────
    print(f"Simulating train  (sim_rand_ic, seq_len={args.train_seq_len} "
          f"= {args.train_seq_len * 0.5:.0f} s, batches={args.train_batches}, "
          f"u_sd=3.0) …")
    np.random.seed(0)
    sim = msd_chain(N=4, T=5000, Ts=0.5, u_sd=3.0, period=args.period)
    u_tr, y_tr = sim.sim_rand_ic(seq_len=args.train_seq_len,
                                 batches=args.train_batches)
    train_in, train_out = sequential_windows(u_tr, y_tr, args.train_seq_len)
    with open(os.path.join(OUT_DIR, args.train_out), "wb") as f:
        pickle.dump((train_in, train_out), f)
    print(f"  → train windows: {train_in.shape}  "
          f"u[std={train_in.std():.3f} rng=[{train_in.min():.2f},{train_in.max():.2f}]]  "
          f"y[std={train_out.std():.3f} rng=[{train_out.min():.2f},{train_out.max():.2f}]]")

    # ── Validation: simulate(T), u_sd=3.0, zero IC ──────────────────────
    if args.skip_val:
        print("Skipping validation regeneration (--skip-val).")
    else:
        print(f"Simulating val    (simulate, T={args.val_T}, u_sd=3.0, zero IC) …")
        np.random.seed(2)
        sim = msd_chain(N=4, T=args.val_T, Ts=0.5, u_sd=3.0, period=args.period)
        u_va, y_va = sim.simulate()
        val_in, val_out = sequential_windows(u_va, y_va, args.val_window)
        with open(os.path.join(OUT_DIR, "msd_data_val.pkl"), "wb") as f:
            pickle.dump((val_in, val_out), f)
        print(f"  → val   windows: {val_in.shape}  "
              f"u[std={val_in.std():.3f} rng=[{val_in.min():.2f},{val_in.max():.2f}]]  "
              f"y[std={val_out.std():.3f} rng=[{val_out.min():.2f},{val_out.max():.2f}]]")

        # Clean (noiseless) validation reference: identical input/IC with v_sd=0
        # so y carries no measurement noise. Re-seeding reproduces the same u as
        # msd_data_val.pkl, keeping the (u, y_clean) pairs index-aligned.
        np.random.seed(2)
        sim_clean = msd_chain(N=4, T=args.val_T, Ts=0.5, u_sd=3.0,
                              period=args.period)
        sim_clean.v_sd = 0.0
        u_vc, y_vc = sim_clean.simulate()
        valc_in, valc_out = sequential_windows(u_vc, y_vc, args.val_window)
        with open(os.path.join(OUT_DIR, "msd_data_val_clean.pkl"), "wb") as f:
            pickle.dump((valc_in, valc_out), f)
        print(f"  → val   clean  : {valc_out.shape}  (noiseless reference)")

    print()
    print(f"Datasets saved to:  {OUT_DIR}/")
    print("Done.")
