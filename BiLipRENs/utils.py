from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np

def l2_norm(x, eps=jnp.finfo(jnp.float32).eps, **kwargs):
    """Compute l2 norm of a vector/matrix with JAX.
    This is safe for backpropagation, unlike `jnp.linalg.norm`."""
    return jnp.sqrt(jnp.maximum(jnp.sum(x**2, **kwargs), eps))

def cayley(W):
    # W in shape n x 2n (m=2n)
    # W = [G H]
    m, n = W.shape 
    if n > m:
       return cayley(W.T).T
    
    G, H = W[:n, :], W[n:, :]

    # Z = GT-G + HTH -------- Eq6
    Z = (G - G.T) + (H.T @ H)
    I = jnp.eye(n)
    Zi = jnp.linalg.inv(I+Z)

    # (I+Z)(I-z)-1    -2V(I-Z)-1
    return jnp.concatenate([Zi @ (I-Z), -2 * H @ Zi], axis=0)

def identity_init():
    """Initialize a weight as the identity matrix.
    
    Assumes that shape is a tuple (n,n), only uses first element.
    """
    def init(key, shape, dtype):
        return jnp.identity(shape[0], dtype)
    return init

def _prepare_indices(n, key, shuffle):
    if shuffle:
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, n)
        return key, indices
    return key, jnp.arange(n)


def _iter_batches(indices, batch_size, drop_last):
    n = indices.shape[0]
    end = n - (n % batch_size) if drop_last else n
    for i in range(0, end, batch_size):
        yield indices[i:i + batch_size]

def _is_ragged(data):
    if isinstance(data, (list, tuple)):
        return True
    return isinstance(data, np.ndarray) and data.dtype == object


def _dataset_len(data):
    return len(data) if isinstance(data, (list, tuple)) else data.shape[0]


def _batch_select(data, batch_indices, padding_value=0):
    if _is_ragged(data):
        batch = [data[int(i)] for i in batch_indices]
        return pad_sequences_in_batch(batch, padding_value=padding_value)
    return data[batch_indices]


def data_generator(
    training_in,
    training_out,
    batch_size,
    epochs,
    key,
    shuffle=True,
    drop_last=False,
    bucket_boundaries=None,
    padding_value=0,
    x0=None,
):
    """
    Unified data generator for fixed-length or ragged sequences.
    - training_out can be None for single-input datasets.
    - x0 can be provided for partial PL training.
    - bucket_boundaries enables length-based bucketing for ragged inputs.
    """
    n = _dataset_len(training_in)
    use_bucket = bucket_boundaries is not None or _is_ragged(training_in)

    if not use_bucket:
        for epoch in range(epochs):
            key, indices = _prepare_indices(n, key, shuffle)
            for batch_idx, batch_indices in enumerate(_iter_batches(indices, batch_size, drop_last)):
                batch_in = training_in[batch_indices]
                batch_out = training_out[batch_indices] if training_out is not None else None
                if x0 is None:
                    if batch_out is None:
                        yield epoch, batch_idx, batch_in
                    else:
                        yield epoch, batch_idx, batch_in, batch_out
                else:
                    yield epoch, batch_idx, batch_in, batch_out, x0[batch_indices]
        return

    grouped_indices = defaultdict(list)
    if bucket_boundaries is None:
        for i, seq in enumerate(training_in):
            grouped_indices[len(seq)].append(i)
    else:
        boundaries = np.array(bucket_boundaries)
        for i, seq in enumerate(training_in):
            bucket_idx = np.searchsorted(boundaries, len(seq))
            grouped_indices[bucket_idx].append(i)

    bucket_indices = {k: jnp.array(v) for k, v in grouped_indices.items()}

    for epoch in range(epochs):
        all_batches = []
        for _, idxs in bucket_indices.items():
            key, bucket_perm = _prepare_indices(idxs.shape[0], key, shuffle)
            bucket_idx = idxs[bucket_perm]
            for batch_indices in _iter_batches(bucket_idx, batch_size, drop_last):
                all_batches.append(batch_indices)

        n_batches = len(all_batches)
        if shuffle:
            key, subkey = jax.random.split(key)
            order = jax.random.permutation(subkey, n_batches)
        else:
            order = jnp.arange(n_batches)

        for batch_idx, order_idx in enumerate(order):
            batch_indices = all_batches[int(order_idx)]
            batch_in = _batch_select(training_in, batch_indices, padding_value=padding_value)
            batch_out = None
            if training_out is not None:
                batch_out = _batch_select(training_out, batch_indices, padding_value=padding_value)
            if x0 is None:
                if batch_out is None:
                    yield epoch, batch_idx, batch_in
                else:
                    yield epoch, batch_idx, batch_in, batch_out
            else:
                yield epoch, batch_idx, batch_in, batch_out, _batch_select(x0, batch_indices, padding_value=padding_value)



def pad_sequences_in_batch(batch_sequences, padding_value=0):
    max_len = max(len(seq) for seq in batch_sequences)
    feature_dim = batch_sequences[0].shape[1] if batch_sequences[0].ndim > 1 else 1
    padded_batch = np.full((len(batch_sequences), max_len, feature_dim), padding_value, dtype=np.float32)
    for i, seq in enumerate(batch_sequences):
        seq_len = len(seq)
        padded_batch[i, :seq_len] = seq.reshape(seq_len, feature_dim)  
    return jnp.array(padded_batch)


def l2_norm_metric(y_true, y_pred, time_axis=None, reduce="mean", eps=1e-12):
    """
    General L2 norm metric for 2D or 3D arrays.
    - 2D: (batch, dim) or (n, dim) -> L2 over dim.
    - 3D: (time, batch, dim) -> L2 over dim, then sum over time.
    Use time_axis to override default if needed.
    """
    err = y_true - y_pred
    per_step = jnp.sum(jnp.square(err), axis=-1)

    if time_axis is None:
        time_axis = 0 if per_step.ndim >= 2 else None

    if time_axis is None:
        per_traj = jnp.sqrt(jnp.maximum(per_step, eps))
    else:
        per_traj = jnp.sqrt(jnp.maximum(jnp.sum(per_step, axis=time_axis), eps))

    if reduce == "none":
        return per_traj
    if reduce == "sum":
        return jnp.sum(per_traj)
    if reduce == "mean":
        return jnp.mean(per_traj)
    raise ValueError(f"Unknown reduce='{reduce}'. Use 'mean', 'sum', or 'none'.")


def l2_norm_loss(y_true, y_pred):
    return l2_norm_metric(y_true, y_pred, time_axis=0, reduce="mean")


def normalize_to_unit(z, z_min, z_max, eps=1e-12):
    """
    Normalize z to [-1, 1]
    """
    return 2.0 * (z - z_min) / (z_max - z_min + eps) - 1.0


def create_states(batch_size, state_size, num_layers, dyn_orth=False, dyn_mult=1):
    """
    Create initial states for scan-based models.

    Returns a list of per-layer state containers compatible with CompREN/ORTHREN.
    """
    ren_state = jnp.zeros((batch_size, state_size))
    if not dyn_orth:
        return [[ren_state] for _ in range(num_layers)]

    orth_state = jnp.zeros((batch_size, dyn_mult * state_size))
    states = [[ren_state, orth_state]]
    for _ in range(1, num_layers):
        states.append([ren_state])
    return states


# ===========================================================================
# Device selection helpers (merged from the former BiLipRENs/device.py)
# ===========================================================================
# Use ONE of these to pick the JAX backend (CPU vs GPU). The selection MUST
# happen before the first JAX computation in the process, otherwise JAX will
# have already initialised on the default backend.
#
# Priority (highest first):
#   1. Explicit argument to ``configure_device(mode=...)`` from Python.
#   2. CLI flag handled by ``add_device_cli_arg`` + ``configure_device_from_args``.
#   3. Environment variable ``BILIPREN_DEVICE`` in {"auto", "cpu", "gpu"}.
#   4. Auto-detect: GPU if available, otherwise CPU.
#
# Typical usage at the top of a script (before importing jax.numpy):
#
#     from BiLipRENs.utils import configure_device
#     configure_device()                    # auto

DeviceMode = Literal["auto", "cpu", "gpu"]
_VALID = ("auto", "cpu", "gpu")

_configured: Optional[str] = None  # records the resolved backend ("cpu"/"gpu")


def _detect_gpu() -> bool:
    """Return True if JAX is able to see a GPU device."""
    try:
        import jax  # local import: do not force jax import at module load time

        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def configure_device(
    mode: Optional[DeviceMode] = None,
    *,
    verbose: bool = True,
) -> str:
    """Configure JAX to use CPU or GPU.

    Parameters
    ----------
    mode
        ``"auto"`` (or ``None``): pick GPU if available, otherwise CPU.
        ``"cpu"``: force CPU even when a GPU is present.
        ``"gpu"``: require GPU; raises RuntimeError if unavailable.
    verbose
        If True (default), print the resolved backend.

    Returns
    -------
    str
        The resolved backend name, ``"cpu"`` or ``"gpu"``.
    """
    global _configured

    # Resolve mode
    if mode is None:
        mode = os.environ.get("BILIPREN_DEVICE", "auto").lower()  # type: ignore[assignment]
    mode = str(mode).lower()  # type: ignore[assignment]
    if mode not in _VALID:
        raise ValueError(f"Unknown device mode {mode!r}; expected one of {_VALID}.")

    # Apply BEFORE jax.numpy imports anywhere; safe to call once.
    if mode == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax

        jax.config.update("jax_platform_name", "cpu")
        resolved = "cpu"
    elif mode == "gpu":
        # Clear any earlier override
        os.environ.pop("JAX_PLATFORMS", None)
        import jax

        try:
            jax.config.update("jax_platform_name", "gpu")
        except Exception:
            pass
        if not _detect_gpu():
            raise RuntimeError(
                "configure_device(mode='gpu') was requested but no GPU is "
                "visible to JAX. Install a CUDA-enabled jaxlib, e.g.\n"
                "    uv pip install -U 'jax[cuda12]'\n"
                "or fall back to mode='cpu'/'auto'."
            )
        resolved = "gpu"
    else:  # auto
        import jax

        try:
            if _detect_gpu():
                resolved = "gpu"
            else:
                os.environ["JAX_PLATFORMS"] = "cpu"
                jax.config.update("jax_platform_name", "cpu")
                resolved = "cpu"
        except Exception:
            os.environ["JAX_PLATFORMS"] = "cpu"
            import jax  # noqa: F811

            jax.config.update("jax_platform_name", "cpu")
            resolved = "cpu"

    _configured = resolved
    if verbose:
        import jax  # noqa: F811

        print(f"[BiLipRENs] device mode={mode!r}  →  backend={resolved}  "
              f"devices={jax.devices()}")
    return resolved


def add_device_cli_arg(parser: argparse.ArgumentParser) -> None:
    """Register a ``--device {auto,cpu,gpu}`` flag on the given parser."""
    parser.add_argument(
        "--device",
        choices=list(_VALID),
        default=None,
        help=(
            "Compute backend for JAX. Default reads $BILIPREN_DEVICE "
            "(falls back to 'auto')."
        ),
    )


def configure_device_from_args(args: argparse.Namespace, **kwargs) -> str:
    """Apply the ``--device`` flag parsed by ``add_device_cli_arg``."""
    return configure_device(getattr(args, "device", None), **kwargs)