"""
eval/evaluate.py
================
Offline evaluation of the exported ONNX model.

Outputs (saved to eval/results/):
  metrics_overall.csv  – accuracy, macro P/R/F1, mean latency
  metrics_per_class.csv – per-class precision, recall, F1, support
  metrics_per_snr.csv   – accuracy in 2 dB SNR bins (-10 to 20)
  confusion_matrix.png
  accuracy_vs_snr.png

Usage:
    # Evaluate on the held-out test set (Pluto session 2):
    python eval/evaluate.py --data pluto_frames.h5 --session 2

    # Evaluate on the held-out TEST groups of the simulated dataset
    # (same --seed/--val-frac/--test-frac as train.py):
    python eval/evaluate.py --data sim/dataset_sim.h5

    # Generalization test on over-the-air captures:
    python eval/evaluate.py --data overair_frames.h5
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Optional

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

with open(_ROOT / "labels.json", encoding="utf-8") as _lf:
    _LDATA = json.load(_lf)
CLASS_NAMES: list[str] = _LDATA["labels"]
N_CLASSES: int = len(CLASS_NAMES)

FRAME_LEN: int = 1024
SNR_BIN_WIDTH: float = 2.0
SNR_MIN: float = -10.0
SNR_MAX: float = 20.0


# ════════════════════════════════════════════════════════════════════════════ #
#  Pre-processing (same as service/inference_service.py)                       #
# ════════════════════════════════════════════════════════════════════════════ #

def _preprocess_batch(X_raw: np.ndarray) -> np.ndarray:
    """
    X_raw: float32 (B, 2, 1024)
    Returns float32 (B, 2, 1024) after DC removal + unit-power normalisation.
    """
    # Remove DC per frame and per channel
    X = X_raw - X_raw.mean(axis=2, keepdims=True)
    # Unit power: mean over both channels and all 1024 samples
    pwr = np.mean(X ** 2, axis=(1, 2), keepdims=True)   # (B, 1, 1)
    pwr = np.maximum(pwr, 1e-12)
    return (X / np.sqrt(pwr)).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════ #
#  Evaluation                                                                  #
# ════════════════════════════════════════════════════════════════════════════ #

def evaluate(
    onnx_path: pathlib.Path,
    data_path: pathlib.Path,
    session_filter: Optional[int],
    batch_size: int,
    out_dir: pathlib.Path,
    split: str = "test",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"[eval] Loading {data_path} …")
    with h5py.File(data_path, "r") as hf:
        if session_filter is not None:
            mask = hf["session"][:] == session_filter
            idx = np.where(mask)[0]
            print(f"[eval] Session {session_filter}: {len(idx):,} frames")
        elif split == "test" and "group" in hf:
            # Recover the held-out test groups exactly as train.py split them
            from train.train import split_by_group
            valid = ~np.isnan(hf["snr_db"][:])
            _tr, _va, idx = split_by_group(hf["group"][:], val_frac, test_frac,
                                           seed, valid_mask=valid)
            print(f"[eval] Held-out TEST split (by waveform group, seed={seed}): "
                  f"{len(idx):,} frames; excluded {len(_tr):,} train + {len(_va):,} val")
        else:
            if split == "test":
                print("[eval] WARNING: no 'group' dataset, cannot recover the held-out "
                      "split; evaluating ALL frames (not held-out unless this file is "
                      "an independent capture).")
            idx = np.arange(hf["X"].shape[0])
            print(f"[eval] All frames: {len(idx):,}")

        X_all   = hf["X"][idx]                     # (N, 2, 1024)
        y_true  = hf["y"][idx].astype(np.int64)   # (N,)
        snr_all = hf["snr_db"][idx].astype(np.float32)  # (N,)

    N = len(y_true)

    # ── ONNX session ──────────────────────────────────────────────────────
    print(f"[eval] Loading ONNX model: {onnx_path}")
    ort_sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    in_name  = ort_sess.get_inputs()[0].name   # "iq"
    out_name = ort_sess.get_outputs()[0].name  # "probs"

    # ── Inference in batches ──────────────────────────────────────────────
    y_pred   = np.empty(N, dtype=np.int64)
    probs_all = np.empty((N, N_CLASSES), dtype=np.float32)
    latencies = np.empty(N, dtype=np.float64)

    print(f"[eval] Running inference on {N:,} frames …")
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X_batch = _preprocess_batch(X_all[start:end])

        t0 = time.perf_counter()
        probs = ort_sess.run([out_name], {in_name: X_batch})[0]   # (B, 11)
        elapsed = time.perf_counter() - t0

        y_pred[start:end] = np.argmax(probs, axis=1)
        probs_all[start:end] = probs
        latencies[start:end] = elapsed / (end - start)   # per-frame latency

    mean_lat_ms = float(np.mean(latencies) * 1000.0)
    overall_acc  = float(np.mean(y_pred == y_true))
    print(f"[eval] Overall accuracy: {overall_acc:.4f}   "
          f"Mean latency: {mean_lat_ms:.3f} ms/frame")

    # ── Per-class metrics ─────────────────────────────────────────────────
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(N_CLASSES), zero_division=0
    )
    df_class = pd.DataFrame({
        "class":     CLASS_NAMES,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "support":   support,
    })
    df_class.to_csv(out_dir / "metrics_per_class.csv", index=False)
    print(df_class.to_string(index=False))

    # ── Overall summary ───────────────────────────────────────────────────
    macro_p = float(np.mean(prec))
    macro_r = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))
    df_overall = pd.DataFrame([{
        "n_frames": N,
        "accuracy": overall_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "mean_latency_ms": mean_lat_ms,
    }])
    df_overall.to_csv(out_dir / "metrics_overall.csv", index=False)

    # ── Per-SNR accuracy (skip NaN snr_db values) ─────────────────────────
    valid_snr = ~np.isnan(snr_all)
    bin_edges = np.arange(SNR_MIN, SNR_MAX + SNR_BIN_WIDTH, SNR_BIN_WIDTH)
    snr_rows = []
    for lo in bin_edges[:-1]:
        hi = lo + SNR_BIN_WIDTH
        mask = valid_snr & (snr_all >= lo) & (snr_all < hi)
        if not np.any(mask):
            continue
        acc_bin = float(np.mean(y_pred[mask] == y_true[mask]))
        snr_rows.append({
            "snr_lo": lo, "snr_hi": hi,
            "n": int(np.sum(mask)),
            "accuracy": acc_bin,
        })

    df_snr = pd.DataFrame(snr_rows)
    df_snr.to_csv(out_dir / "metrics_per_snr.csv", index=False)

    # ── Confusion matrix plot ─────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASSES))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (N={N:,}, acc={overall_acc:.3f})")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if v > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=130)
    plt.close()

    # ── Accuracy vs SNR plot ──────────────────────────────────────────────
    if len(df_snr) > 1:
        valid = df_snr.dropna(subset=["snr_lo"])
        if not valid.empty:
            snr_mids = (valid["snr_lo"] + valid["snr_hi"]) / 2.0
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(snr_mids, valid["accuracy"], "o-", color="#1f77b4", lw=1.5)
            ax2.fill_between(snr_mids, valid["accuracy"], alpha=0.15)
            ax2.set_xlabel("SNR (dB)")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Classification Accuracy vs. SNR")
            ax2.set_ylim(0, 1.05)
            ax2.grid(True, linewidth=0.5, alpha=0.5)
            plt.tight_layout()
            plt.savefig(out_dir / "accuracy_vs_snr.png", dpi=130)
            plt.close()

    print(f"[eval] Results saved to {out_dir}/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ONNX classifier on an HDF5 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",   type=pathlib.Path,
                   default=_ROOT / "model" / "classifier_v1.onnx")
    p.add_argument("--data",    type=pathlib.Path,
                   default=_ROOT / "sim" / "dataset_sim.h5")
    p.add_argument("--session", type=int, default=None,
                   help="Filter by session number (e.g. 2 for Pluto test set)")
    p.add_argument("--split", choices=["test", "all"], default="test",
                   help="'test' = held-out groups only (default); 'all' = every frame "
                        "(use for independent files such as Pluto/over-the-air)")
    p.add_argument("--val-frac",  type=float, default=0.1, help="must match train.py")
    p.add_argument("--test-frac", type=float, default=0.1, help="must match train.py")
    p.add_argument("--seed",      type=int,   default=42,  help="must match train.py")
    p.add_argument("--batch",   type=int, default=512)
    p.add_argument("--out-dir", type=pathlib.Path,
                   default=_ROOT / "eval" / "results")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        onnx_path=args.model,
        data_path=args.data,
        session_filter=args.session,
        batch_size=args.batch,
        out_dir=args.out_dir,
        split=args.split,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
