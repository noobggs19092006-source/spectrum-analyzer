"""build_pluto_dataset.py - cut labeled 1024-sample frames from the conducted captures.

Usage: python build_pluto_dataset.py 1 2
Writes C:\spectrum_rf\pluto_frames.h5 (session 1 for training, session 2 for testing).
"""
import json
import pathlib
import sys

import h5py
import numpy as np
from scipy.signal import resample_poly

FS_RTL = 2_400_000   # RTL-SDR capture rate
FS = 1_600_000       # contract rate
FRAME = 1024
HOP = 4096           # keeps file size manageable
MARGIN_S = 1.0       # drop 1 s at each end of a class window (timing uncertainty)
BASE = pathlib.Path(r"C:\spectrum_rf")
CARRIER_OFFSET_HZ = 0.0   # measured AM-DSB carrier offset (carrier_check.py)


def read_start_ms(path):
    raw = path.read_bytes()
    for enc in ("utf-16", "ascii"):
        try:
            return int(raw.decode(enc).strip().lstrip("\ufeff"))
        except (UnicodeDecodeError, ValueError):
            pass
    raise ValueError(f"Could not read start time from {path}")


def iq_slice(raw, a, b):
    chunk = np.asarray(raw[2 * a: 2 * b], dtype=np.float32)
    x = np.empty(b - a, dtype=np.complex64)
    x.real = (chunk[0::2] - 127.5) / 127.5
    x.imag = (chunk[1::2] - 127.5) / 127.5
    return x


def preprocess(f):
    f = f - f.mean()
    f = f / np.sqrt(np.mean(np.abs(f) ** 2))
    return np.stack([f.real, f.imag]).astype(np.float32)


def main(sessions):
    X, y, s = [], [], []
    for n in sessions:
        d = BASE / f"session{n}"
        r0 = read_start_ms(d / "rtl_start_ms.txt") / 1000.0
        manifest = [json.loads(line) for line in open(d / f"manifest_session_{n}.jsonl")]
        raw = np.memmap(d / f"session{n}_all.bin", dtype=np.uint8, mode="r")

        for m in manifest:
            a = int(round((m["t_start"] - r0 + MARGIN_S) * FS_RTL))
            b = int(round((m["t_end"] - r0 - MARGIN_S) * FS_RTL))
            xs = iq_slice(raw, a, b)
            t = np.arange(len(xs)) / FS_RTL
            xs = (xs * np.exp(-2j * np.pi * CARRIER_OFFSET_HZ * t)).astype(np.complex64)
            y_res = resample_poly(xs, 2, 3)   # 2.4 -> 1.6 MS/s
            count = 0
            for start in range(0, len(y_res) - FRAME, HOP):
                X.append(preprocess(y_res[start:start + FRAME]))
                y.append(m["class_idx"])
                s.append(n)
                count += 1
            print(f"session {n}  {m['class_name']:6s}  frames={count}")

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    s = np.array(s, dtype=np.int32)
    print(f"\nTotal frames: {len(y)}  X shape: {X.shape}")

    out = BASE / "pluto_frames.h5"
    with h5py.File(out, "w") as f:
        f.attrs["fs"] = FS
        f.create_dataset("X", data=X, compression="lzf")
        f.create_dataset("y", data=y)
        f.create_dataset("snr_db", data=np.full(len(y), np.nan, dtype=np.float32))
        f.create_dataset("session", data=s)
        f.create_dataset("source", data=np.ones(len(y), dtype=np.int8))
    print(f"Saved {out}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])


