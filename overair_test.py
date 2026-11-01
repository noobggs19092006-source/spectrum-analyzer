"""overair_test.py - run the classifier on an over-the-air recording and print the class histogram."""
import sys
import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly

LABELS = ["BPSK","QPSK","8PSK","QAM16","QAM64","GFSK","CPFSK","PAM4","WBFM","AMDSB","AMSSB"]
FRAME, HOP = 1024, 4096
MODEL = r"C:\spectrum_rf\model_cfo15\classifier_cfo15.onnx"

raw = np.fromfile(sys.argv[1], dtype=np.uint8)
x = np.empty(len(raw) // 2, dtype=np.complex64)
x.real = (raw[0::2].astype(np.float32) - 127.5) / 127.5
x.imag = (raw[1::2].astype(np.float32) - 127.5) / 127.5
y = resample_poly(x, 2, 3)   # 2.4 -> 1.6 MS/s, same as the contract

frames = []
for s in range(0, len(y) - FRAME, HOP):
    f = y[s:s + FRAME]
    f = f - f.mean()
    f = f / np.sqrt(np.mean(np.abs(f) ** 2))
    frames.append(np.stack([f.real, f.imag]).astype(np.float32))
X = np.stack(frames)

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
probs = np.concatenate([sess.run(None, {name: X[i:i + 4096]})[0] for i in range(0, len(X), 4096)])
counts = np.bincount(probs.argmax(1), minlength=11)

print(f"frames: {len(X)}")
for lab, c in zip(LABELS, counts):
    print(f"  {lab:6s} {c:7d}  {100 * c / len(X):5.1f}%")

