"""am_check.py - sideband asymmetry per class window in session 2 (Pluto)."""
import json
import pathlib
import numpy as np

FS = 2_400_000
NFFT = 4096
d = pathlib.Path(r"C:\spectrum_rf\session2")
raw_ms = (d / "rtl_start_ms.txt").read_bytes()
for enc in ("utf-16", "ascii"):
    try:
        r0 = int(raw_ms.decode(enc).strip().lstrip("\ufeff")) / 1000.0
        break
    except (UnicodeDecodeError, ValueError):
        pass
manifest = [json.loads(l) for l in open(d / "manifest_session_2.jsonl")]
raw = np.memmap(d / "session2_all.bin", dtype=np.uint8, mode="r")
freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / FS))
up_band = (freqs > 10_000) & (freqs < 150_000)
low_band = (freqs < -10_000) & (freqs > -150_000)

print(f"{'class':7s} {'up/low dB':>10s}  (near 0 = two sidebands, large = one sideband)")
for m in manifest:
    a = int((m["t_start"] - r0 + 1.0) * FS)
    b = a + 4 * FS
    c = np.asarray(raw[2 * a: 2 * b], dtype=np.float32)
    x = (c[0::2] - 127.5) / 127.5 + 1j * (c[1::2] - 127.5) / 127.5
    n = len(x) // NFFT
    frames = x[: n * NFFT].reshape(n, NFFT) * np.hanning(NFFT)
    psd = (np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2).mean(axis=0)
    up = psd[up_band].mean()
    low = psd[low_band].mean()
    print(f"{m['class_name']:7s} {10 * np.log10(up / low):10.1f}")
