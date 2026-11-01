"""
radio_gen/pluto_tx.py
=====================
Transmit all 11 modulation classes via PlutoSDR (pyadi-iio).
Works on Windows, macOS, and Linux (pathlib used throughout).

Usage:
    python pluto_tx.py --session 1 --tx-gain -20 --seconds 10
    python pluto_tx.py --session 1 --tx-gain -20 --seconds 10 --dry-run

Requirements (hardware path only):
    pip install pyadi-iio

Dry-run does NOT require pyadi-iio.  It saves:
  tx_output/session_N_dryrun_<CLASS>.npy
  tx_output/session_N_dryrun_<CLASS>.png   (constellation + spectrum)
and still writes:
  tx_output/manifest_session_N.jsonl

NOTE: The Pluto is driven with a cyclic buffer, so each class loops
indefinitely until the sleep interval ends, then the buffer is destroyed
before the next class is loaded.

Check pyadi-iio release notes for attribute-name changes if sdr.tx_lo or
sdr.tx_hardwaregain_chan0 raise AttributeError with future library versions.
"""

from __future__ import annotations
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); import winiio
import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")           # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

# ── make radio_gen importable when run from any working directory ─────────────
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from radio_gen.modgen import (  # noqa: E402
    CLASS_NAMES, CLASS_IDX, FS, SYMBOL_RATE, SPS, gen_waveform,
)

# ── Hardware constants ────────────────────────────────────────────────────────
PLUTO_URI: str = "ip:192.168.2.1"
TX_LO_HZ: int = 915_000_000         # 915 MHz – conducted only, never radiated
TX_RF_BW_HZ: int = 2_000_000        # ~2 MHz RF bandwidth
TX_BUF_LEN: int = 8_192             # cyclic buffer length (must be multiple of 4)

# Pluto DAC scaling: target peak amplitude = 0.5 × full-scale, then × 2^14
_PEAK_FRAC: float = 0.5
_DAC_SCALE: int = 2 ** 14


# ═══════════════════════════════════════════════════════════════════════════ #
#  Waveform preparation                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def _seed_for(session: int, cls_idx: int) -> int:
    """Deterministic, unique seed per (session, class)."""
    return session * 1_000 + cls_idx


def _build_tx_buffer(class_name: str, seed: int) -> np.ndarray:
    """
    Generate one cyclic-buffer worth of IQ samples scaled for the Pluto DAC.

    Returns complex64 array of length TX_BUF_LEN with values in
    approx ±(0.5 × 2^14) ≈ ±8192.
    """
    rng = np.random.default_rng(seed)
    wave = gen_waveform(class_name, TX_BUF_LEN, rng)   # unit power, complex64

    peak = float(np.max(np.abs(wave)))
    if peak > 0.0:
        wave = wave * (_PEAK_FRAC / peak)               # peak at 0.5 FS

    return (wave * _DAC_SCALE).astype(np.complex64)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Dry-run mode                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def _dry_run(session: int, tx_gain_db: float, seconds: int,
             out_dir: pathlib.Path) -> None:
    """Write .npy files + plots and manifest without touching any hardware."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DRY-RUN] session={session}  tx_gain={tx_gain_db} dB  "
          f"duration={seconds}s  out={out_dir}")

    manifest_path = out_dir / f"manifest_session_{session}.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            seed = _seed_for(session, cls_idx)
            buf = _build_tx_buffer(cls_name, seed)

            # ── Save raw samples ──────────────────────────────────────────
            npy_path = out_dir / f"session_{session}_dryrun_{cls_name}.npy"
            np.save(npy_path, buf)

            # ── Plots ─────────────────────────────────────────────────────
            fig, (ax_iq, ax_psd) = plt.subplots(1, 2, figsize=(11, 4))
            fig.suptitle(
                f"Session {session} | {cls_name} | seed={seed}", fontsize=11
            )

            # Constellation (first 2048 samples)
            view = buf[:2048]
            ax_iq.scatter(view.real, view.imag, s=1.5, alpha=0.4, linewidths=0)
            ax_iq.set_title("Constellation")
            ax_iq.set_xlabel("I")
            ax_iq.set_ylabel("Q")
            ax_iq.set_aspect("equal")
            ax_iq.grid(True, linewidth=0.5)

            # Power spectrum (whole buffer)
            N = len(buf)
            freqs_hz = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / FS))
            psd = np.abs(np.fft.fftshift(np.fft.fft(buf))) ** 2
            psd_db = 10.0 * np.log10(psd + 1e-12)
            ax_psd.plot(freqs_hz / 1e3, psd_db, lw=0.6)
            ax_psd.set_title("Power Spectrum")
            ax_psd.set_xlabel("Frequency offset (kHz)")
            ax_psd.set_ylabel("Power (dB, arb.)")
            ax_psd.grid(True, linewidth=0.5)

            plt.tight_layout()
            png_path = out_dir / f"session_{session}_dryrun_{cls_name}.png"
            plt.savefig(png_path, dpi=110)
            plt.close(fig)

            print(f"  [{cls_idx:2d}] {cls_name:8s}  npy={npy_path.name}  "
                  f"png={png_path.name}")

            # ── Manifest entry ─────────────────────────────────────────────
            t_now = time.time()
            record = {
                "session":     session,
                "class_idx":   cls_idx,
                "class_name":  cls_name,
                "t_start":     t_now,
                "t_end":       t_now + seconds,
                "tx_gain_db":  tx_gain_db,
                "fs":          FS,
                "symbol_rate": SYMBOL_RATE,
                "sps":         SPS,
                "center_hz":   TX_LO_HZ,
                "seed":        seed,
                "dry_run":     True,
            }
            mf.write(json.dumps(record) + "\n")

    print(f"[DRY-RUN] Done.  Manifest → {manifest_path}")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Live hardware mode                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

def _live_run(session: int, tx_gain_db: float, seconds: int,
              out_dir: pathlib.Path) -> None:
    """Transmit all 11 classes sequentially via PlutoSDR."""
    try:
        import adi  # pyadi-iio
    except ImportError:
        sys.exit(
            "ERROR: pyadi-iio not found.  Install with:  pip install pyadi-iio\n"
            "       Or use --dry-run to test without hardware."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_session_{session}.jsonl"

    sdr = None
    try:
        print(f"[TX] Connecting to Pluto at {PLUTO_URI} …")
        sdr = adi.Pluto(uri=PLUTO_URI)

        # ── Radio configuration ───────────────────────────────────────────
        # Attribute names current as of pyadi-iio 0.0.17.
        # If a future version renames them, adjust here.
        sdr.sample_rate = FS
        sdr.tx_lo = TX_LO_HZ
        sdr.tx_rf_bandwidth = TX_RF_BW_HZ
        sdr.tx_hardwaregain_chan0 = float(tx_gain_db)   # negative = attenuation
        sdr.tx_cyclic_buffer = True

        print(
            f"[TX] LO={TX_LO_HZ / 1e6:.3f} MHz  BW={TX_RF_BW_HZ / 1e6:.1f} MHz  "
            f"gain={tx_gain_db} dBm"
        )

        with open(manifest_path, "w", encoding="utf-8") as mf:
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                seed = _seed_for(session, cls_idx)
                buf = _build_tx_buffer(cls_name, seed)

                sdr.tx(buf)                         # loads cyclic buffer
                t_start = time.time()
                ts_iso = datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat()
                print(f"  [{cls_idx:2d}] {cls_name:8s}  TX {seconds}s  "
                      f"seed={seed}  t={ts_iso}")

                time.sleep(seconds)
                t_end = time.time()

                # Stop cyclic buffer before loading next class
                sdr.tx_destroy_buffer()

                record = {
                    "session":     session,
                    "class_idx":   cls_idx,
                    "class_name":  cls_name,
                    "t_start":     t_start,
                    "t_end":       t_end,
                    "tx_gain_db":  tx_gain_db,
                    "fs":          FS,
                    "symbol_rate": SYMBOL_RATE,
                    "sps":         SPS,
                    "center_hz":   TX_LO_HZ,
                    "seed":        seed,
                    "dry_run":     False,
                }
                mf.write(json.dumps(record) + "\n")

    finally:
        # Always release hardware, even on exception
        if sdr is not None:
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
            try:
                del sdr
            except Exception:
                pass
        print("[TX] Device released.")

    print(f"[TX] Done.  Manifest → {manifest_path}")


# ═══════════════════════════════════════════════════════════════════════════ #
#  CLI entry point                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transmit 11 modulation classes via PlutoSDR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--session", type=int, required=True,
                   help="Session number (written to manifest and HDF5)")
    p.add_argument("--tx-gain", type=float, default=-30.0,
                   help="TX hardware gain in dB (negative = attenuation)")
    p.add_argument("--seconds", type=int, default=10,
                   help="Transmit duration per class (seconds)")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip hardware; write .npy and plots only")
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path(".") / "tx_output",
        help="Output directory for manifest and dry-run artefacts",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dry_run:
        _dry_run(args.session, args.tx_gain, args.seconds, args.out_dir)
    else:
        _live_run(args.session, args.tx_gain, args.seconds, args.out_dir)


if __name__ == "__main__":
    main()
