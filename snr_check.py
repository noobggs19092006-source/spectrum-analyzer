"""snr_check.py - per-class SNR and clipping for a conducted capture.

Usage:  python snr_check.py 1
        python snr_check.py 2
"""
import argparse
import csv
import json
import pathlib

import numpy as np

FS = 2_400_000          # RTL-SDR sample rate
NFFT = 4096
BAND_HZ = 300_000       # signals sit at DC (915 MHz LO); measure +/-300 kHz
EDGE_S = 0.5            # skip this much at each end of a class window
SIM_BW_HZ = 1_600_000   # simulator's full noise bandwidth (1.6 MS/s)


def read_start_ms(path):
    raw = pathlib.Path(path).read_bytes()
    for enc in ("utf-16", "ascii"):
        try:
            return int(raw.decode(enc).strip().lstrip("\ufeff"))
        except (UnicodeDecodeError, ValueError):
            pass
    raise ValueError(f"Could not read start time from {path}")


def load_iq(path):
    raw = np.fromfile(path, dtype=np.uint8)
    x = np.empty(len(raw) // 2, dtype=np.complex64)
    x.real = (raw[0::2].astype(np.float32) - 127.5) / 127.5
    x.imag = (raw[1::2].astype(np.float32) - 127.5) / 127.5
    return x


def in_band_power(seg):
    n = len(seg) // NFFT
    if n == 0:
        raise ValueError("segment too short")
    frames = seg[: n * NFFT].reshape(n, NFFT) * np.hanning(NFFT).astype(np.float32)
    psd = (np.abs(np.fft.fft(frames, axis=1)) ** 2).mean(axis=0)
    f = np.fft.fftfreq(NFFT, 1 / FS)
    return float(psd[np.abs(f) <= BAND_HZ].mean())


def clip_fraction(seg):
    return float(np.mean((np.abs(seg.real) >= 0.99) | (np.abs(seg.imag) >= 0.99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=int)
    args = ap.parse_args()
    n = args.session
    d = pathlib.Path(rf"C:\spectrum_rf\session{n}")

    x = load_iq(d / f"session{n}_all.bin")
    r0 = read_start_ms(d / "rtl_start_ms.txt") / 1000.0
    manifest = [json.loads(line) for line in open(d / f"manifest_session_{n}.jsonl")]

    def idx(t):
        return max(0, int(round((t - r0) * FS)))

    # Noise floor from the last 10 s: transmissions end at ~128 s, so the tail is clean
    p_noise = in_band_power(x[-10 * FS:])
    noise_db = 10 * np.log10(p_noise)
    sim_offset = 10 * np.log10(SIM_BW_HZ / (2 * BAND_HZ))

    rows = []
    for m in manifest:
        a, b = idx(m["t_start"] + EDGE_S), idx(m["t_end"] - EDGE_S)
        seg = x[a:b]
        p_sig = in_band_power(seg)
        snr_in = 10 * np.log10(max(p_sig - p_noise, 1e-12) / p_noise)
        rows.append({
            "class_name": m["class_name"],
            "tx_gain_db": m["tx_gain_db"],
            "snr_inband_db": round(snr_in, 2),
            "snr_sim_equiv_db": round(snr_in - sim_offset, 2),
            "clip_pct": round(100 * clip_fraction(seg), 3),
        })

    print(f"\nSession {n}  noise floor (in-band): {noise_db:.1f} dB")
    print(f"{'class':7s}{'TX dB':>8s}{'SNR in':>10s}{'SNR sim':>10s}{'clip %':>10s}")
    for r in rows:
        print(f"{r['class_name']:7s}{r['tx_gain_db']:8.1f}{r['snr_inband_db']:10.1f}"
              f"{r['snr_sim_equiv_db']:10.1f}{r['clip_pct']:10.3f}")

    out = d / f"snr_session_{n}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
