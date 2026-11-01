"""
service/inference_service.py
=============================
Subscribe to frames from the teammate's frame cutter (port 5556),
classify with the ONNX model, and publish results (port 5557).

ZMQ contract (spec §4):
  SUB  tcp://127.0.0.1:5556  topic b"frames"
       multipart: [b"frames", header_json_bytes, payload_bytes]
       header:  {"ts", "frame_id", "center_hz", "bw_hz", "power_dbfs", "fs"}
       payload: 2048 × float32 little-endian  (I row then Q row)

  PUB  tcp://127.0.0.1:5557  topic b"results"
       multipart: [b"results", json_bytes]
       JSON: {"ts", "frame_id", "center_hz", "bw_hz",
              "class", "class_idx", "probs"[11], "snr_est", "anomaly_flag"}

Usage:
    python service/inference_service.py --model model/classifier_v1.onnx
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import onnxruntime as ort
import zmq

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from service.anomaly import AnomalyDetector   # noqa: E402

# ── Label map ────────────────────────────────────────────────────────────────
with open(_ROOT / "labels.json", encoding="utf-8") as _lf:
    _LABELS = json.load(_lf)
IDX_TO_LABEL: dict[int, str] = {
    int(k): v for k, v in _LABELS["idx_to_label"].items()
}

FRAME_LEN: int = 1024
FS: int = 1_600_000

SUB_ADDR: str = "tcp://127.0.0.1:5556"
PUB_ADDR: str = "tcp://127.0.0.1:5557"


# ════════════════════════════════════════════════════════════════════════════ #
#  Pre-processing (must match sim_gen.py exactly)                              #
# ════════════════════════════════════════════════════════════════════════════ #

def _preprocess(x_iq: np.ndarray) -> np.ndarray:
    """Remove DC, normalise to unit power. Returns float32 (1, 2, 1024)."""
    x_iq = x_iq - np.mean(x_iq)
    arr = np.stack([x_iq.real, x_iq.imag], axis=0)  # (2, 1024)
    pwr = float(np.mean(arr ** 2))
    if pwr > 0.0:
        arr = arr / math.sqrt(pwr)
    return arr[np.newaxis].astype(np.float32)        # (1, 2, 1024)


# ════════════════════════════════════════════════════════════════════════════ #
#  SNR estimator (M2M4)                                                        #
# ════════════════════════════════════════════════════════════════════════════ #

def _estimate_snr_db(x_iq: np.ndarray) -> float:
    """Rough moment-based SNR estimate (M2M4 method).

    Returns NaN when the estimate is undefined (discriminant 2*M2^2 - M4 <= 0,
    or the signal power is zero) instead of a misleading floor value.
    """
    m2 = float(np.mean(np.abs(x_iq) ** 2))
    m4 = float(np.mean(np.abs(x_iq) ** 4))
    disc = 2.0 * m2 ** 2 - m4
    if m2 <= 0.0 or disc <= 0.0:
        return float("nan")
    denom = m2 - math.sqrt(disc)
    if denom <= 0.0:
        return float("nan")
    snr_lin = math.sqrt(disc) / denom
    if snr_lin <= 0.0:
        return float("nan")
    return float(10.0 * math.log10(snr_lin))


# ════════════════════════════════════════════════════════════════════════════ #
#  Main service loop                                                           #
# ════════════════════════════════════════════════════════════════════════════ #

def run(
    model_path: pathlib.Path,
    verbose: bool = False,
    max_frames: int | None = None,
) -> None:
    # ── ONNX session ──────────────────────────────────────────────────────
    print(f"[inference] Loading ONNX model: {model_path}")
    ort_sess = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    input_name  = ort_sess.get_inputs()[0].name   # "iq"
    output_name = ort_sess.get_outputs()[0].name  # "probs"

    anomaly = AnomalyDetector()

    # ── ZMQ setup ─────────────────────────────────────────────────────────
    ctx = zmq.Context()

    sub = ctx.socket(zmq.SUB)
    sub.connect(SUB_ADDR)
    sub.setsockopt(zmq.SUBSCRIBE, b"frames")
    sub.setsockopt(zmq.RCVTIMEO, 5000)   # 5 s timeout so Ctrl-C works

    pub = ctx.socket(zmq.PUB)
    pub.bind(PUB_ADDR)
    time.sleep(0.1)   # allow subscribers to connect

    print(f"[inference] SUB → {SUB_ADDR}   PUB ← {PUB_ADDR}")
    print("[inference] Waiting for frames … (Ctrl-C to stop)")

    n_frames = 0
    latency_sum = 0.0

    try:
        while True:
            try:
                parts = sub.recv_multipart()
            except zmq.Again:
                continue   # timeout – loop so Ctrl-C is caught

            if len(parts) != 3:
                continue   # malformed message

            _topic, header_bytes, payload_bytes = parts

            try:
                header = json.loads(header_bytes.decode("utf-8"))
            except Exception:
                continue

            # ── Decode payload ────────────────────────────────────────────
            raw = np.frombuffer(payload_bytes, dtype="<f4")   # little-endian float32
            if raw.size != 2 * FRAME_LEN:
                continue
            x_iq = raw[:FRAME_LEN] + 1j * raw[FRAME_LEN:]    # (1024,) complex

            # ── Inference ────────────────────────────────────────────────
            t0 = time.perf_counter()
            x_in = _preprocess(x_iq)                          # (1, 2, 1024)
            probs = ort_sess.run([output_name], {input_name: x_in})[0][0]  # (11,)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            cls_idx = int(np.argmax(probs))
            cls_name = IDX_TO_LABEL[cls_idx]
            snr_est = _estimate_snr_db(x_iq)

            # ── Anomaly check ─────────────────────────────────────────────
            power_dbfs = float(header.get("power_dbfs", 0.0))
            center_hz  = int(header.get("center_hz", 0))
            anomaly_flag = anomaly.update(center_hz, power_dbfs)

            # ── Publish result ────────────────────────────────────────────
            result = {
                "ts":           header.get("ts", time.time()),
                "frame_id":     header.get("frame_id", n_frames),
                "center_hz":    center_hz,
                "bw_hz":        header.get("bw_hz", FS),
                "class":        cls_name,
                "class_idx":    cls_idx,
                "probs":        probs.tolist(),
                "snr_est":      None if math.isnan(snr_est) else snr_est,
                "anomaly_flag": bool(anomaly_flag),
            }
            pub.send_multipart([b"results", json.dumps(result).encode("utf-8")])

            n_frames += 1
            latency_sum += latency_ms

            if max_frames is not None and n_frames >= max_frames:
                print(f"[inference] Reached max frames ({max_frames}). Stopping.")
                break

            if verbose or n_frames % 100 == 0:
                avg_lat = latency_sum / n_frames
                print(
                    f"[{n_frames:6d}] {cls_name:8s}  conf={probs[cls_idx]:.3f}  "
                    f"snr={snr_est:+.1f} dB  anomaly={anomaly_flag}  "
                    f"lat={latency_ms:.1f}ms (avg {avg_lat:.1f}ms)"
                )

    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        pub.close()
        ctx.term()
        if n_frames > 0:
            print(
                f"\n[inference] Processed {n_frames} frames. "
                f"Mean latency: {latency_sum / n_frames:.2f} ms"
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modulation classification service")
    p.add_argument("--model", type=pathlib.Path,
                   default=_ROOT / "model" / "classifier_v1.onnx")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Exit after processing N frames")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.model, verbose=args.verbose, max_frames=args.max_frames)
