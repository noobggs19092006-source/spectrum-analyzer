"""power_timeline.py - power per second for a session recording,
with the class that the manifest says is active at each second.

Usage: python power_timeline.py 1
"""
import json
import pathlib
import sys

import numpy as np

FS = 2_400_000

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


r0 = read_start_ms(d / "rtl_start_ms.txt") / 1000.0
manifest = [json.loads(line) for line in open(d / f"manifest_session_{n}.jsonl")]

raw = np.memmap(d / f"session{n}_all.bin", dtype=np.uint8, mode="r")
n_blocks = len(raw) // (2 * FS)

print(f"Session {n}: {n_blocks} seconds")
print(" sec    power   active class")
for s in range(n_blocks):
    chunk = np.asarray(raw[2 * s * FS: 2 * (s + 1) * FS], dtype=np.float32)
    i = (chunk[0::2] - 127.5) / 127.5
    q = (chunk[1::2] - 127.5) / 127.5
    p_db = 10 * np.log10(np.mean(i * i + q * q) + 1e-20)

    t = s + 0.5
    tag = ""
    for m in manifest:
        if m["t_start"] - r0 <= t <= m["t_end"] - r0:
            tag = m["class_name"]
    print(f"{s:4d}  {p_db:7.1f} dB  {tag}")
