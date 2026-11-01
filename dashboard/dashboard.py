"""
dashboard/dashboard.py
=======================
Live modulation-classification dashboard.

Subscribes to  tcp://127.0.0.1:5557  topic b"results"
(published by service/inference_service.py).

Shows:
  • Scrolling timeline of classified bursts (class label + confidence)
  • Live bar chart of the last frame's class probabilities
  • Anomaly alert row  (red = active)

Run:
    python dashboard/dashboard.py

Requires: matplotlib (already in requirements.txt)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
# Select interactive or headless backend gracefully
if os.environ.get("HEADLESS") == "1" or "--headless" in sys.argv:
    matplotlib.use("Agg")
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import zmq

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Label list ────────────────────────────────────────────────────────────────
with open(_ROOT / "labels.json", encoding="utf-8") as _lf:
    _LDATA = json.load(_lf)
CLASS_NAMES: list[str] = _LDATA["labels"]   # ordered 0-10
N_CLASSES = len(CLASS_NAMES)

SUB_ADDR: str = "tcp://127.0.0.1:5557"
HISTORY: int = 60        # bursts to show in timeline
POLL_MS: int = 50        # ZMQ poll timeout per animation tick
UPDATE_EVERY: int = 1    # redraw every N received frames

# ── Colour palette (one per class) ────────────────────────────────────────────
_PALETTE = plt.cm.tab20(np.linspace(0, 1, N_CLASSES))


# ════════════════════════════════════════════════════════════════════════════ #
#  Dashboard state                                                             #
# ════════════════════════════════════════════════════════════════════════════ #

class _State:
    def __init__(self) -> None:
        self.cls_history:   collections.deque = collections.deque(maxlen=HISTORY)
        self.conf_history:  collections.deque = collections.deque(maxlen=HISTORY)
        self.anom_history:  collections.deque = collections.deque(maxlen=HISTORY)
        self.snr_history:   collections.deque = collections.deque(maxlen=HISTORY)
        self.last_probs: list[float] = [1.0 / N_CLASSES] * N_CLASSES
        self.total_frames: int = 0
        self.last_ts: float = time.time()


# ════════════════════════════════════════════════════════════════════════════ #
#  Main                                                                        #
# ════════════════════════════════════════════════════════════════════════════ #

def run(
    max_frames: int | None = None,
    headless: bool = False,
    snapshot_path: pathlib.Path | None = None,
) -> None:
    # ── ZMQ ──────────────────────────────────────────────────────────────
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(SUB_ADDR)
    sub.setsockopt(zmq.SUBSCRIBE, b"results")
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    state = _State()

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 7), facecolor="#0e1117")
    fig.suptitle("Wireless Spectrum Analyzer — Modulation Dashboard",
                 color="white", fontsize=13, y=0.97)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.07, right=0.97,
                           top=0.91, bottom=0.08,
                           hspace=0.45, wspace=0.35)

    ax_timeline  = fig.add_subplot(gs[0, :])   # full-width timeline
    ax_probs     = fig.add_subplot(gs[1, 0])   # probability bars
    ax_snr       = fig.add_subplot(gs[1, 1])   # SNR estimate history
    ax_anomaly   = fig.add_subplot(gs[2, :])   # anomaly flag strip

    _style_axis(ax_timeline, "Classification Timeline",
                "Burst index", "Class")
    _style_axis(ax_probs,    "Class Probabilities (last frame)",
                "Class", "Probability")
    _style_axis(ax_snr,      "SNR Estimate (dB)",
                "Burst index", "SNR (dB)")
    _style_axis(ax_anomaly,  "Anomaly Flag (per burst)",
                "Burst index", "")

    if not headless:
        plt.ion()
        plt.show()

    print("[dashboard] Ready. Waiting for results...", flush=True)
    frame_since_redraw = 0

    try:
        while True:
            if max_frames is not None and state.total_frames >= max_frames:
                break

            # ── Non-blocking poll ─────────────────────────────────────────
            events = dict(poller.poll(POLL_MS))
            if sub in events:
                try:
                    parts = sub.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    parts = []

                if len(parts) == 2:
                    try:
                        result = json.loads(parts[1].decode("utf-8"))
                    except Exception:
                        result = None

                    if result is not None:
                        _ingest(state, result)
                        frame_since_redraw += 1

                        if frame_since_redraw >= UPDATE_EVERY:
                            _redraw(state, ax_timeline, ax_probs,
                                    ax_snr, ax_anomaly)
                            if not headless:
                                fig.canvas.draw_idle()
                                fig.canvas.flush_events()
                            frame_since_redraw = 0
            else:
                if not headless:
                    fig.canvas.flush_events()
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        if snapshot_path is not None:
            _redraw(state, ax_timeline, ax_probs, ax_snr, ax_anomaly)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(snapshot_path, dpi=120, facecolor=fig.get_facecolor())
            print(f"[dashboard] Saved snapshot → {snapshot_path}")
        sub.close()
        ctx.term()
        if not headless:
            plt.ioff()
        plt.close("all")
        print(f"\n[dashboard] Exiting. Total frames shown: {state.total_frames}")


# ════════════════════════════════════════════════════════════════════════════ #
#  Helpers                                                                     #
# ════════════════════════════════════════════════════════════════════════════ #

def _style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.set_title(title, color="#c9d1d9", fontsize=8, pad=4)
    ax.set_xlabel(xlabel, color="#8b949e", fontsize=7)
    ax.set_ylabel(ylabel, color="#8b949e", fontsize=7)


def _ingest(state: _State, result: dict) -> None:
    cls_idx = int(result.get("class_idx", 0))
    probs   = result.get("probs", [1.0 / N_CLASSES] * N_CLASSES)
    conf    = float(probs[cls_idx])
    anom    = bool(result.get("anomaly_flag", False))
    snr_raw = result.get("snr_est", None)
    snr     = float("nan") if snr_raw is None else float(snr_raw)

    state.cls_history.append(cls_idx)
    state.conf_history.append(conf)
    state.anom_history.append(anom)
    state.snr_history.append(snr)
    state.last_probs = probs
    state.total_frames += 1


def _redraw(state: _State,
            ax_timeline, ax_probs, ax_snr, ax_anomaly) -> None:
    cls_arr  = list(state.cls_history)
    conf_arr = list(state.conf_history)
    anom_arr = list(state.anom_history)
    snr_arr  = list(state.snr_history)
    xs = list(range(len(cls_arr)))

    # ── Timeline scatter ─────────────────────────────────────────────────
    ax_timeline.cla()
    _style_axis(ax_timeline, "Classification Timeline", "Burst index", "Class")
    if cls_arr:
        colors = [_PALETTE[c] for c in cls_arr]
        sizes  = [20 + 80 * c for c in conf_arr]
        ax_timeline.scatter(xs, cls_arr, c=colors, s=sizes, alpha=0.85,
                            linewidths=0, zorder=3)
        ax_timeline.set_yticks(range(N_CLASSES))
        ax_timeline.set_yticklabels(CLASS_NAMES, fontsize=6, color="white")
        ax_timeline.set_ylim(-0.5, N_CLASSES - 0.5)
        ax_timeline.grid(True, axis="x", color="#30363d", linewidth=0.5)

    # ── Probability bars ──────────────────────────────────────────────────
    ax_probs.cla()
    _style_axis(ax_probs, "Class Probabilities (last frame)", "Class", "P")
    probs = state.last_probs
    bar_colors = [_PALETTE[i] for i in range(N_CLASSES)]
    ax_probs.bar(range(N_CLASSES), probs, color=bar_colors, width=0.7)
    ax_probs.set_xticks(range(N_CLASSES))
    ax_probs.set_xticklabels(CLASS_NAMES, rotation=45, ha="right",
                              fontsize=6, color="white")
    ax_probs.set_ylim(0, 1.05)
    ax_probs.axhline(1.0 / N_CLASSES, color="#8b949e", lw=0.8, ls="--")

    # ── SNR history ───────────────────────────────────────────────────────
    ax_snr.cla()
    _style_axis(ax_snr, "SNR Estimate (dB)", "Burst index", "dB")
    finite_snr = [v for v in snr_arr if np.isfinite(v)]
    if finite_snr:
        # NaN = estimate undefined for that frame; leave a gap in the line
        ax_snr.plot(xs, snr_arr, color="#58a6ff", lw=1.0)
        ax_snr.fill_between(xs, snr_arr, alpha=0.15, color="#58a6ff")
        ax_snr.set_ylim(min(finite_snr) - 3, max(finite_snr) + 3)

    # ── Anomaly strip ─────────────────────────────────────────────────────
    ax_anomaly.cla()
    _style_axis(ax_anomaly, "Anomaly Flag", "Burst index", "")
    if anom_arr:
        anom_colors = ["#f85149" if a else "#238636" for a in anom_arr]
        ax_anomaly.bar(xs, [1] * len(xs), color=anom_colors,
                       width=1.0, align="center")
        ax_anomaly.set_yticks([])
        n_active = sum(anom_arr)
        label = (f"⚠  {n_active} anomalous burst(s) in window"
                 if n_active else "✓  No anomalies in window")
        color = "#f85149" if n_active else "#3fb950"
        ax_anomaly.set_title(
            f"Anomaly Flag  —  {label}",
            color=color, fontsize=8, pad=4,
        )
    ax_anomaly.set_ylim(0, 1.5)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live modulation dashboard")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Exit automatically after N frames (useful for testing)")
    p.add_argument("--headless", action="store_true",
                   help="Run headless without opening a display window")
    p.add_argument("--snapshot", type=pathlib.Path, default=None,
                   help="Save snapshot image on exit")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        max_frames=args.max_frames,
        headless=args.headless,
        snapshot_path=args.snapshot,
    )
