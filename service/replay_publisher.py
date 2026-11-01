"""
service/replay_publisher.py
============================
Read a dataset HDF5 file and publish frames in the same ZMQ format
that the teammate's live frame cutter uses (port 5556, topic b"frames").

This lets the full Python pipeline be tested without any SDR hardware.

Usage:
    python service/replay_publisher.py --data sim/dataset_sim.h5
    python service/replay_publisher.py --data pluto_frames.h5 --rate 100
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import h5py
import numpy as np
import zmq

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FRAME_LEN: int = 1024
FS: int = 1_600_000
PUB_ADDR: str = "tcp://127.0.0.1:5556"


def replay(
    h5_path: pathlib.Path,
    rate_hz: float,
    loop: bool,
    max_frames: int | None,
    anomaly_at: int | None = None,
) -> None:
    """
    Publish frames from an HDF5 file at the requested rate.

    Parameters
    ----------
    h5_path   : path to dataset HDF5 (must have X, y, snr_db)
    rate_hz   : frames per second  (0 = as fast as possible)
    loop      : restart from beginning when EOF is reached
    max_frames: stop after this many frames (None = unlimited)
    """
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(PUB_ADDR)
    time.sleep(1.0)   # pause so subscribers establish connection

    print(f"[replay] Publishing from {h5_path}  →  {PUB_ADDR}")
    print(f"[replay] Rate: {'unlimited' if rate_hz <= 0 else f'{rate_hz:.1f} fps'}  "
          f"Loop: {loop}")

    period = (1.0 / rate_hz) if rate_hz > 0 else 0.0
    frame_id = 0

    try:
        with h5py.File(h5_path, "r") as hf:
            N = hf["X"].shape[0]
            y = hf["y"][:]
            snr_db = hf["snr_db"][:]

            while True:
                for i in range(N):
                    if max_frames is not None and frame_id >= max_frames:
                        return

                    t_start = time.perf_counter()

                    # Load one frame
                    x_np = hf["X"][i]                      # float32 (2, 1024)
                    x_iq = x_np[0] + 1j * x_np[1]         # complex (1024,)

                    # Real stored frame power if available, otherwise compute from samples
                    if "power_dbfs" in hf:
                        power_dbfs = float(hf["power_dbfs"][i])
                    else:
                        pwr = float(np.mean(np.abs(x_iq) ** 2))
                        power_dbfs = float(10.0 * np.log10(max(pwr, 1e-12)))

                    if anomaly_at is not None and anomaly_at <= frame_id < anomaly_at + 3:
                        power_dbfs += 25.0

                    # Synthetic centre frequency (all sim at 915 MHz)
                    center_hz = 915_000_000

                    # ── Header ────────────────────────────────────────────
                    header = {
                        "ts":          time.time(),
                        "frame_id":    frame_id,
                        "center_hz":   center_hz,
                        "bw_hz":       FS,
                        "power_dbfs":  power_dbfs,
                        "fs":          FS,
                        # These are only present in replay; live headers won't have them
                        "_true_class": int(y[i]),
                        "_snr_db":     float(snr_db[i]) if not np.isnan(snr_db[i]) else None,
                    }
                    header_bytes = json.dumps(header).encode("utf-8")

                    # ── Payload: I row then Q row, little-endian float32 ──
                    payload = np.concatenate([x_np[0], x_np[1]]).astype("<f4")
                    payload_bytes = payload.tobytes()

                    pub.send_multipart([b"frames", header_bytes, payload_bytes])

                    frame_id += 1
                    if frame_id % 500 == 0:
                        print(f"[replay] Sent {frame_id} frames")

                    # ── Rate limiting ──────────────────────────────────────
                    if period > 0:
                        elapsed = time.perf_counter() - t_start
                        sleep_for = period - elapsed
                        if sleep_for > 0:
                            time.sleep(sleep_for)

                if not loop:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        pub.close()
        ctx.term()
        print(f"[replay] Sent {frame_id} frames total.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay HDF5 frames as ZMQ stream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=pathlib.Path,
                   default=_ROOT / "sim" / "dataset_sim.h5")
    p.add_argument("--rate", type=float, default=50.0,
                   help="Frames per second (0 = unlimited)")
    p.add_argument("--loop", action="store_true",
                   help="Loop the dataset indefinitely")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many frames")
    p.add_argument("--anomaly-at", type=int, default=None,
                   help="Inject a 3-burst power anomaly starting at this frame_id")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    replay(
        h5_path=args.data,
        rate_hz=args.rate,
        loop=args.loop,
        max_frames=args.max_frames,
        anomaly_at=args.anomaly_at,
    )
