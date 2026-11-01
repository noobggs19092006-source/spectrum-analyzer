"""
sim/sim_gen.py
==============
Generate 300 000 simulated training frames and save to dataset_sim.h5.

Output schema (spec Â§2):
  X:       float32  (N, 2, 1024)  â€“ row 0 = I, row 1 = Q
  y:       int64    (N,)
  snr_db:  float32  (N,)
  group:   int32    (N,)   waveform group id (used for leakage-free splits)
  power_dbfs: float32 (N,) SYNTHETIC power (see note in _generate_class_frames)
  session: int32    (N,)   all 0  (simulation)
  source:  int8     (N,)   all 0  (sim = 0)
  file attribute  fs = 1 600 000

Preprocessing applied to EVERY frame (same as inference):
  1. Subtract complex mean  (remove DC)
  2. Divide by sqrt(mean(|x|^2))  (unit average power)

Augmentations applied before preprocessing:
  - AWGN at SNR ~ Uniform(-10, 20) dB
  - Carrier frequency offset ~ Uniform(-5, +5) kHz
  - Random phase offset ~ Uniform(0, 2Ï€)
  - Fractional timing offset ~ Uniform(-0.5, +0.5) samples
  - 2-tap multipath  (direct + 1 reflected tap)
  - IQ amplitude / phase imbalance
  - 8-bit (256-level) ADC quantisation of I and Q

Usage:
    python sim/sim_gen.py                        # 300 000 frames
    python sim/sim_gen.py --frames 10000         # quick smoke-test
    python sim/sim_gen.py --workers 4            # parallel (macOS/Linux)
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import pathlib
import sys
import time

import h5py
import numpy as np
from tqdm import tqdm

# â”€â”€ make radio_gen importable when run as a script â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from radio_gen.modgen import CLASS_NAMES, FS, gen_waveform  # noqa: E402

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#  Configuration                                                               #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #

N_CLASSES: int = len(CLASS_NAMES)   # 11
FRAME_LEN: int = 1024               # samples per frame
GEN_LEN: int = 8192                 # samples generated per waveform call
FRAMES_PER_GEN: int = 6             # frames cut from each generation
MASTER_SEED: int = 42               # change to get a different dataset version

DEFAULT_N_FRAMES: int = 300_000
DEFAULT_OUT: pathlib.Path = _ROOT / "sim" / "dataset_sim.h5"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#  Pre-processing (must match inference_service.py exactly)                    #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #

def preprocess(x_iq: np.ndarray) -> np.ndarray:
    """
    Apply the fixed preprocessing contract to one complex IQ frame.

    Parameters
    ----------
    x_iq : complex array, shape (FRAME_LEN,)

    Returns
    -------
    float32 array, shape (2, FRAME_LEN) â€“ row 0 = I, row 1 = Q,
    mean removed, unit average power.
    """
    x_iq = x_iq - np.mean(x_iq)                     # remove DC
    iq = np.stack([x_iq.real, x_iq.imag], axis=0)   # (2, 1024)
    pwr = float(np.mean(iq ** 2))
    if pwr > 0.0:
        iq = iq / math.sqrt(pwr)
    return iq.astype(np.float32)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#  Augmentation helpers                                                        #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #

def _add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN so that signal-to-noise ratio equals snr_db."""
    sig_pwr = float(np.mean(np.abs(x) ** 2))
    noise_pwr = sig_pwr / (10.0 ** (snr_db / 10.0))
    std = math.sqrt(noise_pwr / 2.0)           # per-component std
    noise = std * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return (x + noise).astype(np.complex64)


def _apply_cfo(x: np.ndarray, freq_hz: float) -> np.ndarray:
    """Rotate by carrier frequency offset freq_hz (Hz)."""
    t = np.arange(len(x), dtype=np.float64)
    rot = np.exp(1j * 2.0 * math.pi * freq_hz / FS * t, dtype=np.complex128)
    return (x * rot).astype(np.complex64)


def _apply_phase(x: np.ndarray, phi_rad: float) -> np.ndarray:
    return (x * np.exp(1j * phi_rad)).astype(np.complex64)


def _fractional_delay(x: np.ndarray, tau: float) -> np.ndarray:
    """Sub-sample delay via frequency-domain phase ramp."""
    N = len(x)
    f = np.fft.fftfreq(N)                          # normalised frequencies
    X = np.fft.fft(x.astype(np.complex128))
    X_shifted = X * np.exp(-1j * 2.0 * math.pi * f * tau)
    return np.fft.ifft(X_shifted).astype(np.complex64)


def _multipath(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """2-tap multipath: direct path + one reflected tap."""
    amp = rng.uniform(0.0, 0.5)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    delay = int(rng.integers(1, 9))                 # 1â€“8 samples
    tap = amp * np.exp(1j * phi)

    y = x.copy().astype(np.complex64)
    if delay < len(x):
        y[delay:] = y[delay:] + tap * x[:-delay]
    return y


def _iq_imbalance(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    IQ amplitude + phase imbalance.
    amplitude_imbalance: Â±0.5 dB â†’ linear factor on Q
    phase_imbalance:     Â±2Â°  â†’ cross-coupling between I and Q
    """
    amp_db = rng.uniform(-0.5, 0.5)
    amp = 10.0 ** (amp_db / 20.0)                  # linear gain on Q channel
    phi = math.radians(rng.uniform(-2.0, 2.0))     # phase error

    I = x.real.astype(np.float64)
    Q = x.imag.astype(np.float64)
    I_out = I
    Q_out = math.sin(phi) * I + amp * math.cos(phi) * Q
    return (I_out + 1j * Q_out).astype(np.complex64)


def _quantize_8bit(x: np.ndarray) -> np.ndarray:
    """Simulate 8-bit ADC quantisation on I and Q independently."""
    scale = float(np.max(np.abs(x)))
    if scale == 0.0:
        return x
    x_n = x / scale                                  # normalise to Â±1
    # 8-bit signed: 256 levels â†’ resolution 1/127
    I_q = np.round(x_n.real * 127.0) / 127.0
    Q_q = np.round(x_n.imag * 127.0) / 127.0
    return (I_q + 1j * Q_q).astype(np.complex64)


def augment_frame(
    frame: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply all augmentations in physical order, then return complex64."""
    # 1. Fractional timing offset (receiver timing error)
    tau = rng.uniform(-0.5, 0.5)
    frame = _fractional_delay(frame, tau)

    # 2. Multipath channel
    frame = _multipath(frame, rng)

    # 3. Carrier frequency offset
    cfo_hz = rng.uniform(-15e3, 15e3)
    frame = _apply_cfo(frame, cfo_hz)

    # 4. Random phase offset (LO phase at receiver)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    frame = _apply_phase(frame, phi)

    # 5. IQ hardware imbalance
    frame = _iq_imbalance(frame, rng)

    # 6. AWGN (after all analogue impairments, at ADC input)
    frame = _add_awgn(frame, snr_db, rng)

    # 7. 8-bit ADC quantisation
    frame = _quantize_8bit(frame)

    return frame


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#  Per-class frame generation (designed to be called in a subprocess)          #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #

def _generate_class_frames(
    cls_idx: int,
    n_frames: int,
    seed_base: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate n_frames augmented frames for CLASS_NAMES[cls_idx].

    Returns (cls_idx, X_class, snr_class, group_class, pwr_class) where
      X_class    : float32 (n_frames, 2, 1024)
      snr_class  : float32 (n_frames,)
      group_class: int32   (n_frames,)
      pwr_class  : float32 (n_frames,)
    """
    cls_name = CLASS_NAMES[cls_idx]
    rng = np.random.default_rng([seed_base, cls_idx])

    X_out = np.empty((n_frames, 2, FRAME_LEN), dtype=np.float32)
    snr_out = np.empty(n_frames, dtype=np.float32)
    group_out = np.empty(n_frames, dtype=np.int32)
    pwr_out = np.empty(n_frames, dtype=np.float32)

    frame_count = 0
    n_gens = math.ceil(n_frames / FRAMES_PER_GEN)

    for gen_idx in range(n_gens):
        if frame_count >= n_frames:
            break

        # Generate a long waveform; cut several frames from it.
        # All frames from this waveform share the same group ID.
        wave = gen_waveform(cls_name, GEN_LEN, rng)
        group_id = int(cls_idx * 1_000_000 + gen_idx)

        max_start = GEN_LEN - FRAME_LEN
        for _ in range(FRAMES_PER_GEN):
            if frame_count >= n_frames:
                break

            start = int(rng.integers(0, max_start + 1))
            raw_frame = wave[start: start + FRAME_LEN].copy()

            snr_db = float(rng.uniform(-10.0, 20.0))
            aug_frame = augment_frame(raw_frame, snr_db, rng)

            # power_dbfs is SYNTHETIC: a random per-frame baseline (N(-35, 1.5) dB)
            # plus the frame's relative level before preprocess() rescales it to
            # unit power. It is not a measured RF power; it only gives the anomaly
            # detector a plausible baseline to work against in replay tests.
            raw_pwr = float(np.mean(np.abs(aug_frame) ** 2))
            base_pwr_dbfs = float(rng.normal(-35.0, 1.5))
            real_pwr_dbfs = base_pwr_dbfs + float(10.0 * np.log10(max(raw_pwr, 1e-6)))

            X_out[frame_count] = preprocess(aug_frame)
            snr_out[frame_count] = snr_db
            group_out[frame_count] = group_id
            pwr_out[frame_count] = real_pwr_dbfs
            frame_count += 1

    return (
        cls_idx,
        X_out[:frame_count],
        snr_out[:frame_count],
        group_out[:frame_count],
        pwr_out[:frame_count],
    )


def _worker(args: tuple[int, int, int]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multiprocessing-safe wrapper."""
    return _generate_class_frames(*args)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#  Main dataset builder                                                        #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #

def build_dataset(
    n_total: int = DEFAULT_N_FRAMES,
    out_path: pathlib.Path = DEFAULT_OUT,
    workers: int = 1,
    seed: int = MASTER_SEED,
) -> None:
    frames_per_class = n_total // N_CLASSES
    actual_total = frames_per_class * N_CLASSES     # balanced

    print(
        f"[sim_gen] Generating {actual_total:,} frames "
        f"({frames_per_class:,}/class Ã— {N_CLASSES} classes) â€¦"
    )
    print(f"[sim_gen] Master seed: {seed}   Workers: {workers}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # â”€â”€ Collect per-class results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tasks = [(cls_idx, frames_per_class, seed) for cls_idx in range(N_CLASSES)]

    if workers > 1:
        ctx = mp.get_context("spawn")   # safe on macOS and Windows
        with ctx.Pool(processes=workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_worker, tasks),
                    total=N_CLASSES,
                    desc="Classes",
                    unit="class",
                )
            )
    else:
        results = []
        for task in tqdm(tasks, desc="Classes", unit="class"):
            results.append(_generate_class_frames(*task))

    # Sort by class index (imap may not preserve order across workers)
    results.sort(key=lambda r: r[0])

    # â”€â”€ Concatenate and shuffle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_list, y_list, snr_list, grp_list, pwr_list = [], [], [], [], []
    for cls_idx, X_cls, snr_cls, grp_cls, pwr_cls in results:
        X_list.append(X_cls)
        y_list.append(np.full(len(X_cls), cls_idx, dtype=np.int64))
        snr_list.append(snr_cls)
        grp_list.append(grp_cls)
        pwr_list.append(pwr_cls)

    X_all = np.concatenate(X_list, axis=0)        # (N, 2, 1024)
    y_all = np.concatenate(y_list, axis=0)        # (N,)
    snr_all = np.concatenate(snr_list, axis=0)    # (N,)
    grp_all = np.concatenate(grp_list, axis=0)    # (N,)
    pwr_all = np.concatenate(pwr_list, axis=0)    # (N,)

    # Shuffle (deterministic)
    rng_shuf = np.random.default_rng(seed + 1)
    perm = rng_shuf.permutation(len(y_all))
    X_all = X_all[perm]
    y_all = y_all[perm]
    snr_all = snr_all[perm]
    grp_all = grp_all[perm]
    pwr_all = pwr_all[perm]

    # NOTE: no anomaly is written into the dataset. Power spikes are injected
    # at replay time only (service/replay_publisher.py --anomaly-at).
    N = len(y_all)

    session_all = np.zeros(N, dtype=np.int32)
    source_all = np.zeros(N, dtype=np.int8)       # 0 = sim

    # â”€â”€ Save HDF5 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"[sim_gen] Writing {out_path} â€¦")
    t0 = time.perf_counter()
    with h5py.File(out_path, "w") as hf:
        hf.attrs["fs"] = FS
        hf.attrs["master_seed"] = seed
        hf.attrs["n_classes"] = N_CLASSES
        hf.attrs["frame_len"] = FRAME_LEN
        hf.attrs["class_names"] = CLASS_NAMES
        hf.attrs["power_dbfs_note"] = (
            "SYNTHETIC: random N(-35, 1.5) dB baseline + relative frame level; "
            "not a measured power. No anomalies are stored in this file."
        )

        hf.create_dataset("X",          data=X_all,       compression="lzf")
        hf.create_dataset("y",          data=y_all)
        hf.create_dataset("snr_db",     data=snr_all.astype(np.float32))
        hf.create_dataset("group",      data=grp_all.astype(np.int32))
        hf.create_dataset("power_dbfs", data=pwr_all.astype(np.float32))
        hf.create_dataset("session",    data=session_all)
        hf.create_dataset("source",     data=source_all)

    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(
        f"[sim_gen] Done.  {N:,} frames  â†’  {out_path}  "
        f"({size_mb:.1f} MB)  [{elapsed:.1f}s]"
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#  CLI                                                                         #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate simulated training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames", type=int, default=DEFAULT_N_FRAMES,
                   help="Total number of frames (will be rounded to multiple of 11)")
    p.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT,
                   help="Output HDF5 path")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (1 = single-process, safe everywhere)")
    p.add_argument("--seed", type=int, default=MASTER_SEED,
                   help="Master random seed")
    return p.parse_args()


if __name__ == "__main__":
    # Required for multiprocessing safety on macOS / Windows
    mp.freeze_support()
    args = _parse_args()
    build_dataset(
        n_total=args.frames,
        out_path=args.out,
        workers=args.workers,
        seed=args.seed,
    )

