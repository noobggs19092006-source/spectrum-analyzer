r"""live_frame_cutter.py - reads IQ from rtl_tcp, detects bursts, and publishes
1024-sample frames on ZMQ (tcp://127.0.0.1:5556, topic b"frames") in the contract format.

Start rtl_tcp first, in its own window:
    C:\Users\aryan\Downloads\Release\x64\rtl_tcp.exe -a 127.0.0.1 -p 1234

Then in the venv window:
    python live_frame_cutter.py                          # 915 MHz, burst detection
    python live_frame_cutter.py --freq 100.1e6 --continuous   # FM test, cuts every 2 s
"""
import argparse
import json
import socket
import struct
import time

import numpy as np
import zmq
from scipy.signal import resample_poly

RTL_HOST, RTL_PORT = "127.0.0.1", 1234
PUB_ADDR = "tcp://127.0.0.1:5556"
FS_RTL = 2_400_000
FS = 1_600_000
FRAME = 1024
BLOCK = 4096           # detection block, about 1.7 ms
WARMUP_S = 0.5         # drop the startup transient
THRESH_DB = 3.0        # burst threshold above the noise floor
MIN_BURST = 8192       # RTL-rate samples; shorter bursts are ignored
MAX_BURST_S = 2.0      # close long bursts so memory stays bounded
NOISE_BLOCKS = 300     # noise-floor window, about 0.5 s


def connect_rtl(freq_hz, gain_tenths):
    s = socket.create_connection((RTL_HOST, RTL_PORT))
    for c, v in ((0x01, int(freq_hz)), (0x02, FS_RTL), (0x03, 1), (0x04, gain_tenths)):
        s.sendall(struct.pack(">BI", c, v))
    return s


def read_exact(s, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("rtl_tcp closed the connection")
        buf += chunk
    return bytes(buf)


def to_complex(raw_bytes):
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
    x = np.empty(len(raw) // 2, dtype=np.complex64)
    x.real = (raw[0::2] - 127.5) / 127.5
    x.imag = (raw[1::2] - 127.5) / 127.5
    return x


def burst_center_offset(x):
    """Approximate carrier offset in Hz: power-weighted centroid of the spectrum."""
    n = 4096
    seg = x[: len(x) // n * n].reshape(-1, n)
    psd = (np.abs(np.fft.fft(seg * np.hanning(n), axis=1)) ** 2).mean(axis=0)
    psd = np.fft.fftshift(psd)
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / FS_RTL))
    keep = psd > 0.1 * psd.max()
    return float(np.sum(f[keep] * psd[keep]) / np.sum(psd[keep]))


def cut_frames(x, f_off, pub, state, center_hz):
    t = np.arange(len(x)) / FS_RTL
    x = x * np.exp(-2j * np.pi * f_off * t)          # shift the burst to baseband
    y = resample_poly(x, 2, 3)                        # 2.4 MS/s -> 1.6 MS/s
    sent = 0
    for start in range(0, len(y) - FRAME + 1, FRAME):
        f = y[start:start + FRAME]
        p_db = 10 * np.log10(np.mean(np.abs(f) ** 2) + 1e-20)
        payload = np.concatenate([f.real, f.imag]).astype("<f4").tobytes()
        header = {
            "ts": time.time(),
            "frame_id": state["fid"],
            "center_hz": int(round(center_hz + f_off)),
            "bw_hz": FS,
            "power_dbfs": round(float(p_db), 2),
            "fs": FS,
        }
        pub.send_multipart([b"frames", json.dumps(header).encode(), payload])
        state["fid"] += 1
        sent += 1
    return sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, default=915e6)
    ap.add_argument("--gain", type=int, default=297, help="tenths of dB")
    ap.add_argument("--continuous", action="store_true",
                    help="skip detection and cut every 2 s (for testing on a steady signal)")
    args = ap.parse_args()
    center = int(args.freq)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(PUB_ADDR)
    time.sleep(1.0)                      # give subscribers time to connect

    s = connect_rtl(center, args.gain)
    print(f"Tuned to {center/1e6:.3f} MHz, publishing to {PUB_ADDR}"
          + ("  [continuous]" if args.continuous else ""))

    state = {"fid": 0}
    noise_hist, burst, burst_len = [], [], 0
    total_frames = 0
    t_start = time.time()
    try:
        while True:
            x = to_complex(read_exact(s, BLOCK * 2))
            if time.time() - t_start < WARMUP_S:
                continue
            p = float(np.mean(np.abs(x) ** 2))
            if not noise_hist:
                noise_hist.append(p)
                continue
            noise = float(np.median(noise_hist))
            above = args.continuous or p > noise * 10 ** (THRESH_DB / 10)

            if above:
                burst.append(x)
                burst_len += len(x)
                if burst_len < MAX_BURST_S * FS_RTL:
                    continue                  # keep collecting
            else:
                noise_hist.append(p)          # only quiet blocks update the noise floor
                if len(noise_hist) > NOISE_BLOCKS:
                    noise_hist.pop(0)

            if burst:
                seg = np.concatenate(burst)
                burst, burst_len = [], 0
                if len(seg) >= MIN_BURST:
                    f_off = burst_center_offset(seg)
                    n = cut_frames(seg, f_off, pub, state, center)
                    total_frames += n
                    print(f"burst {len(seg)/FS_RTL:.2f}s  center {(center+f_off)/1e6:.4f} MHz"
                          f"  frames {n}  total {total_frames}")
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        s.close()
        pub.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()


