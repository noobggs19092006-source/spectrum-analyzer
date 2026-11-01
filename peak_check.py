import sys
import numpy as np

FS, N = 2_400_000, 65536

raw = np.fromfile(sys.argv[1], dtype=np.uint8).astype(np.float32)
iq = ((raw[0::2] - 127.5) + 1j * (raw[1::2] - 127.5)) / 127.5

win = np.hanning(N)
nseg = len(iq) // N
psd = np.zeros(N)
for k in range(nseg):
    psd += np.abs(np.fft.fftshift(np.fft.fft(iq[k*N:(k+1)*N] * win))) ** 2
psd_db = 10 * np.log10(psd / nseg + 1e-20)
f = np.fft.fftshift(np.fft.fftfreq(N, 1 / FS))

# Ignore the carrier-leakage spike near 0 Hz (within 20 kHz)
mask = np.abs(f) > 20_000
peak_idx = np.argmax(np.where(mask, psd_db, -np.inf))
floor = np.median(psd_db[mask])

print(f"Peak at {f[peak_idx]/1e3:+.1f} kHz from center")
print(f"Peak is {psd_db[peak_idx] - floor:.1f} dB above the noise floor")