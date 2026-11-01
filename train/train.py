"""
train/train.py
==============
Train a VT-CNN2-style 1D CNN for modulation classification.

Architecture summary:
  Input  (B, 2, 1024)
  Conv1d  2→64,  k=7, padding=3  + BN + ReLU + MaxPool(2)  →  (B, 64,  512)
  Conv1d 64→128, k=5, padding=2  + BN + ReLU + MaxPool(2)  →  (B, 128, 256)
  Conv1d 128→128,k=3, padding=1  + BN + ReLU + MaxPool(2)  →  (B, 128, 128)
  Conv1d 128→256,k=3, padding=1  + BN + ReLU               →  (B, 256, 128)
  AdaptiveAvgPool1d(1) + Flatten                             →  (B, 256)
  Linear(256, 128) + ReLU + Dropout(0.3)
  Linear(128, 11)                                            →  logits

≈ 225 000 parameters (well under 1 M).

Usage (simulated data only):
    python train/train.py --data sim/dataset_sim.h5

Usage (simulated + Pluto session 1):
    python train/train.py --data sim/dataset_sim.h5 --pluto pluto_frames.h5
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
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ════════════════════════════════════════════════════════════════════════════ #
#  Dataset                                                                     #
# ════════════════════════════════════════════════════════════════════════════ #

class IQDataset(Dataset):
    """
    Lazy-read HDF5 wrapper.  Keeps the file open for fast random access.

    Parameters
    ----------
    h5_path : path to dataset_sim.h5 or pluto_frames.h5
    indices  : subset row indices (for train/val split)
    pluto_session : if set, keep only rows where session == this value
    """

    def __init__(
        self,
        h5_path: pathlib.Path,
        indices: Optional[np.ndarray] = None,
        pluto_session: Optional[int] = None,
        skip_nan_snr: bool = True,
    ) -> None:
        self.hf = h5py.File(h5_path, "r")
        N = self.hf["X"].shape[0]

        valid_mask = np.ones(N, dtype=bool)
        if skip_nan_snr and "snr_db" in self.hf:
            snr = self.hf["snr_db"][:]
            valid_mask = valid_mask & (~np.isnan(snr))

        if pluto_session is not None:
            sess = self.hf["session"][:]
            valid_mask = valid_mask & (sess == pluto_session)

        base_idx = np.where(valid_mask)[0]
        # Load arrays into RAM once; per-row reads from compressed HDF5 are very slow
        self.X = self.hf["X"][:]
        self.y = self.hf["y"][:]

        if indices is not None:
            # Ensure indices only include valid (non-NaN, matching session) rows
            self.idx = np.intersect1d(base_idx, indices)
        else:
            self.idx = base_idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        row = int(self.idx[i])
        x = torch.from_numpy(self.X[row])           # float32 (2, 1024)
        y = int(self.y[row])
        return x, y


# ════════════════════════════════════════════════════════════════════════════ #
#  Model                                                                       #
# ════════════════════════════════════════════════════════════════════════════ #

class ModulationCNN(nn.Module):
    """VT-CNN2-style 1D CNN for 11-class modulation recognition."""

    def __init__(self, n_classes: int = 11, dropout: float = 0.3) -> None:
        super().__init__()

        def _block(in_ch, out_ch, k, pool=True):
            layers = [
                nn.Conv1d(in_ch, out_ch, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool1d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            _block(2,   64,  7, pool=True),   # →  (B, 64,  512)
            _block(64,  128, 5, pool=True),   # →  (B, 128, 256)
            _block(128, 128, 3, pool=True),   # →  (B, 128, 128)
            _block(128, 256, 3, pool=False),  # →  (B, 256, 128)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          # →  (B, 256,  1)
            nn.Flatten(),                     # →  (B, 256)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════════════════ #
#  Early stopping                                                              #
# ════════════════════════════════════════════════════════════════════════════ #

class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 1e-5) -> None:
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ════════════════════════════════════════════════════════════════════════════ #
#  Training loop                                                               #
# ════════════════════════════════════════════════════════════════════════════ #

def split_by_group(
    groups: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int,
    valid_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic train / val / test split by waveform group ID.

    No waveform group appears in more than one split. val is used for
    checkpoint selection; test is held out and only used by eval/evaluate.py.
    evaluate.py calls this same function with the same seed to recover the
    test rows.
    """
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm_groups = rng.permutation(unique_groups)
    n_val = max(1, int(len(unique_groups) * val_frac))
    n_test = max(1, int(len(unique_groups) * test_frac))
    val_groups = perm_groups[:n_val]
    test_groups = perm_groups[n_val:n_val + n_test]

    is_val = np.isin(groups, val_groups)
    is_test = np.isin(groups, test_groups)
    is_train = ~(is_val | is_test)
    if valid_mask is not None:
        is_val, is_test, is_train = (is_val & valid_mask, is_test & valid_mask,
                                     is_train & valid_mask)
    return np.where(is_train)[0], np.where(is_val)[0], np.where(is_test)[0]


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/val index split (fallback when groups not present)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    return perm[n_val:], perm[:n_val]


def train(
    sim_path: pathlib.Path,
    pluto_path: Optional[pathlib.Path],
    out_dir: pathlib.Path,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    patience: int,
    seed: int,
    test_frac: float = 0.1,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"[train] Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────
    with h5py.File(sim_path, "r") as hf:
        N_sim = hf["X"].shape[0]
        valid_mask = np.ones(N_sim, dtype=bool)
        if "snr_db" in hf:
            snr_sim = hf["snr_db"][:]
            valid_mask = valid_mask & (~np.isnan(snr_sim))
            n_nan = int(np.sum(np.isnan(snr_sim)))
            if n_nan > 0:
                print(f"[train] Skipping {n_nan:,} frames with NaN snr_db in {sim_path}")

        if "group" in hf:
            groups = hf["group"][:]
            train_idx, val_idx, test_idx = split_by_group(
                groups, val_frac, test_frac, seed, valid_mask=valid_mask)
            print(f"[train] Split by waveform group: {len(np.unique(groups)):,} total groups "
                  f"(train/val/test rows: {len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}; "
                  f"test is held out, not used in training)")
        else:
            valid_indices = np.where(valid_mask)[0]
            rng = np.random.default_rng(seed)
            perm = rng.permutation(valid_indices)
            n_val = max(1, int(len(perm) * val_frac))
            n_test = max(1, int(len(perm) * test_frac))
            val_idx = perm[:n_val]
            test_idx = perm[n_val:n_val + n_test]
            train_idx = perm[n_val + n_test:]
            print("[train] WARNING: no 'group' dataset; falling back to a random "
                  "frame-level split (evaluate.py cannot recover it).")

    sim_train = IQDataset(sim_path, indices=train_idx)
    sim_val   = IQDataset(sim_path, indices=val_idx)

    train_ds: Dataset = sim_train
    val_ds:   Dataset = sim_val

    if pluto_path is not None:
        # Pluto session 1 → training only  (session 2 is held-out TEST)
        pluto_train = IQDataset(pluto_path, pluto_session=1, skip_nan_snr=False)
        train_ds = ConcatDataset([sim_train, pluto_train])
        print(f"[train] Merged Pluto session-1 frames: {len(pluto_train):,}")

    print(
        f"[train] Train: {len(train_ds):,}  Val: {len(val_ds):,}  "
        f"Simulated val frames: {len(sim_val):,}"
    )

    n_workers = 0   # h5py file handles are not fork-safe; keep 0 for safety
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=n_workers, pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = ModulationCNN(n_classes=11, dropout=0.3).to(device)
    print(f"[train] Parameters: {model.n_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )
    early_stop = EarlyStopping(patience=patience)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"
    log_path  = out_dir / "train_log.jsonl"

    best_val_loss = float("inf")

    with open(log_path, "w", encoding="utf-8") as log_fh:
        for epoch in range(1, epochs + 1):
            # ── Train ─────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            t0 = time.perf_counter()

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(y_batch)
                train_correct += (logits.argmax(1) == y_batch).sum().item()
                train_total += len(y_batch)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # ── Validate ──────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * len(y_batch)
                    val_correct += (logits.argmax(1) == y_batch).sum().item()
                    val_total += len(y_batch)

            val_loss /= val_total
            val_acc = val_correct / val_total
            scheduler.step()
            elapsed = time.perf_counter() - t0

            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.1f}s]"
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss,   "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            log_fh.write(json.dumps(row) + "\n")
            log_fh.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": best_val_loss,
                        "val_acc": val_acc,
                        "seed": seed,
                    },
                    ckpt_path,
                )

            if early_stop(val_loss):
                print(f"[train] Early stopping at epoch {epoch}.")
                break

    print(f"[train] Best val_loss={best_val_loss:.4f}  Checkpoint → {ckpt_path}")


# ════════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                         #
# ════════════════════════════════════════════════════════════════════════════ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train modulation classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",    type=pathlib.Path,
                   default=_ROOT / "sim" / "dataset_sim.h5",
                   help="Path to simulated HDF5 dataset")
    p.add_argument("--pluto",   type=pathlib.Path, default=None,
                   help="Optional: path to pluto_frames.h5  (session 1 used)")
    p.add_argument("--out-dir", type=pathlib.Path, default=_ROOT / "model",
                   help="Directory to save checkpoints and logs")
    p.add_argument("--epochs",  type=int,   default=60)
    p.add_argument("--batch",   type=int,   default=256)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--val-frac",type=float, default=0.1)
    p.add_argument("--test-frac",type=float, default=0.1,
                   help="Fraction of waveform groups held out for evaluation")
    p.add_argument("--patience",type=int,   default=60,
                   help="Early-stopping patience (default = epochs, i.e. effectively off)")
    p.add_argument("--seed",    type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        sim_path=args.data,
        pluto_path=args.pluto,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        val_frac=args.val_frac,
        patience=args.patience,
        seed=args.seed,
        test_frac=args.test_frac,
    )
