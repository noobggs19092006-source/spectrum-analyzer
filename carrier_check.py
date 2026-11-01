"""carrier_check.py - find the carrier (strongest narrow peak) in each class window.
Usage: python carrier_check.py 2
"""
import json, pathlib, sys
import numpy as np

FS = 2_400_000
NFFT = 65536
n = int(sys.argv[1])
d = pathlib.Path(rf"C:\spectrum_rf\session{n}")

def read_start_ms(path):
    raw = pathlib.Path(path).read_bytes()
    for enc in ("utf-16", "ascii"):
        try:
            return int(raw.decode(enc).strip().lstrip("\ufeff"))
        except (UnicodeDecodeError, ValueError):
            pass
    raise ValueError("bad start time file")

r0 = read_start_ms(d / "rtl_start_ms.txt") / 1000.0
manifest = [json.loads(l) for l in open(d / f"manifest_session_{n}.jsonl")]
raw = np.memmap(d / f"session{n}_all.bin", dtype=np.uint8, mode="r")
freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / FS))
band = np.abs(freqs) < 60_000
win = np.hanning(NFFT)

print(f"{'class':7s} {'peak Hz':>10s} {'prominence dB':>14s}")
for m in manifest:
    a = int((m["t_start"] - r0 + 1.0) * FS)
    c = np.asarray(raw[2 * a: 2 * (a + 2 * FS)], dtype=np.float32)
    x = (c[0::2] - 127.5) / 127.5 + 1j * (c[1::2] - 127.5) / 127.5
    k = len(x) // NFFT
    segs = x[: k * NFFT].reshape(k, NFFT) * win
    psd = (np.abs(np.fft.fftshift(np.fft.fft(segs, axis=1), axes=1)) ** 2).mean(axis=0)
    pb = 10 * np.log10(psd[band])
    i = int(np.argmax(pb))
    print(f"{m['class_name']:7s} {freqs[band][i]:10.0f} {pb[i] - np.median(pb):14.1f}")
