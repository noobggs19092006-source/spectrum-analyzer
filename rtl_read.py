import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\spectrum_rf\test.bin"

raw = np.fromfile(path, dtype=np.uint8)
i = (raw[0::2].astype(np.float32) - 127.5) / 127.5
q = (raw[1::2].astype(np.float32) - 127.5) / 127.5
iq = i + 1j * q

print("Samples:", len(iq))
print("Mean power (dB):", 10 * np.log10(np.mean(np.abs(iq) ** 2)))