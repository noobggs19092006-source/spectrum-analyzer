"""
model/export_onnx.py
====================
Export the trained classifier to ONNX and verify with onnxruntime.

Contract (spec §3E):
  - Input  name : "iq"    shape (batch, 2, 1024)  dynamic batch axis
  - Output name : "probs" shape (batch, 11)        includes softmax
  - Opset 17
  - Max abs error vs PyTorch < 1e-4

Usage:
    python model/export_onnx.py                          # uses model/best_model.pt
    python model/export_onnx.py --ckpt model/best_model.pt --out model/classifier_v1.onnx
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.train import ModulationCNN  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════ #
#  Wrapper that bakes softmax into the graph                                   #
# ════════════════════════════════════════════════════════════════════════════ #

class _ModelWithSoftmax(nn.Module):
    def __init__(self, backbone: ModulationCNN) -> None:
        super().__init__()
        self.backbone = backbone
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.backbone(x))


# ════════════════════════════════════════════════════════════════════════════ #
#  Export + verify                                                             #
# ════════════════════════════════════════════════════════════════════════════ #

def export(ckpt_path: pathlib.Path, out_path: pathlib.Path) -> None:
    device = torch.device("cpu")

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"[export] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    backbone = ModulationCNN(n_classes=11, dropout=0.3)
    backbone.load_state_dict(ckpt["model_state_dict"])
    backbone.eval()

    wrapped = _ModelWithSoftmax(backbone).eval()

    # ── Dummy input ───────────────────────────────────────────────────────
    dummy = torch.randn(4, 2, 1024, dtype=torch.float32)

    # ── Export to ONNX ────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] Exporting to {out_path} (opset 17) …")
    export_kwargs = {
        "opset_version": 17,
        "input_names": ["iq"],
        "output_names": ["probs"],
        "dynamic_axes": {"iq": {0: "batch"}, "probs": {0: "batch"}},
        "do_constant_folding": True,
    }
    try:
        torch.onnx.export(wrapped, dummy, str(out_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapped, dummy, str(out_path), **export_kwargs)

    # ── Validate ONNX graph ───────────────────────────────────────────────
    model_onnx = onnx.load(str(out_path))
    onnx.checker.check_model(model_onnx)
    print("[export] ONNX graph check: PASSED")

    # ── Numerical verification ────────────────────────────────────────────
    ort_sess = ort.InferenceSession(str(out_path),
                                    providers=["CPUExecutionProvider"])
    test_input = torch.randn(8, 2, 1024, dtype=torch.float32)

    with torch.no_grad():
        torch_out = wrapped(test_input).numpy()

    ort_out = ort_sess.run(["probs"], {"iq": test_input.numpy()})[0]

    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    print(f"[export] Max abs diff (PyTorch vs ORT): {max_diff:.2e}")

    if max_diff > 1e-4:
        raise RuntimeError(
            f"Numerical mismatch too large: {max_diff:.2e} > 1e-4"
        )

    print(f"[export] Verification PASSED  ({out_path.stat().st_size / 1024:.1f} KB)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export classifier to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=pathlib.Path,
                   default=_ROOT / "model" / "best_model.pt")
    p.add_argument("--out",  type=pathlib.Path,
                   default=_ROOT / "model" / "classifier_v1.onnx")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(args.ckpt, args.out)
