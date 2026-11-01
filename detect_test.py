"""detect_test.py - run the burst detector offline on a recorded session.

Usage: python detect_test.py 1
Shows how many class windows are detected at several thresholds, and any
detections that fall outside every class window (false alarms).
"""
import json
import pathlib
import sys

import numpy as np

FS = 2_400_000
BLOCK = 2_400          # 1 ms blocks, exact at 2.4 MS/s
SMOOTH = 30            # 30 ms moving average
MIN_RUN = 100          # ignore runs shorter than 100 ms

n = int(sys.argv[1])
d = pathlib.Path(rf"C:\spectrum_rf\session{n}")


def read_start_ms(path):
    raw = pathlib.Path(path).read_bytes()
    for enc in ("utf-16", "ascii"):
        try:
            return int(raw.decode(enc).strip().lstrip("\ufeff"))
        except (UnicodeDecodeError, ValueError):
            pass
    raise ValueError(f"Could not read start time from {path}")


def runs_of(mask):
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


r0 = read_start_ms(d / "rtl_start_ms.txt") / 1000.0
manifest = [json.loads(line) for line in open(d / f"manifest_session_{n}.jsonl")]

raw = np.memmap(d / f"session{n}_all.bin", dtype=np.uint8, mode="r")
pdb = []
for s in range(len(raw) // (2 * FS)):
    c = np.asarray(raw[2 * s * FS: 2 * (s + 1) * FS], dtype=np.float32)
    i = (c[0::2] - 127.5) / 127.5
    q = (c[1::2] - 127.5) / 127.5
    p = (i * i + q * q).reshape(-1, BLOCK).mean(axis=1)
    pdb.append(10 * np.log10(p + 1e-20))
pdb = np.concatenate(pdb)
block_t = np.arange(len(pdb)) * (BLOCK / FS)
noise = float(np.median(pdb[-10000:]))
sm = np.convolve(pdb, np.ones(SMOOTH) / SMOOTH, mode="same")

print(f"Session {n}: noise median {noise:.2f} dB, {len(pdb)} blocks")

for thr in (1, 2, 3, 6):
    mask = sm > noise + thr

    cov = []
    for m in manifest:
        a, b = m["t_start"] - r0, m["t_end"] - r0
        w = (block_t >= a) & (block_t < b)
        cov.append((m["class_name"], float(mask[w].mean()) if w.any() else 0.0))
    hits = sum(c > 0.5 for _, c in cov)

    false = []
    for s, e in runs_of(mask):
        if e - s < MIN_RUN:
            continue
        mid = block_t[s] + (e - s) * (BLOCK / FS) / 2
        if not any(m["t_start"] - r0 <= mid <= m["t_end"] - r0 for m in manifest):
            false.append(round(float(mid), 1))

    print(f"\nthreshold +{thr} dB: {hits}/11 classes detected")
    print("  " + "  ".join(f"{name}:{c:.2f}" for name, c in cov))
    print(f"  detections outside class windows (at seconds): {false}")
