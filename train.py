"""
train.py
========
Training script for SpecCompress-Net with Knowledge Distillation.

Quick start (Google Colab / local)
-----------------------------------
.. code-block:: bash

    python train.py \\
        --real_dir   data/real   \\
        --fake_dir   data/fake   \\
        --output_dir checkpoints \\
        --epochs     30          \\
        --batch_size 32          \\
        --lr         1e-4

Dataset layout expected
-----------------------
``real_dir/`` and ``fake_dir/`` must each contain image frames (PNG / JPG).
Frames can be extracted from videos in advance with ``utils.extract_frames``.

Training strategy
-----------------
1. The Teacher is **fine-tuned only on its last block (layer4)** using the
   uncompressed frames / raw spectra.
2. The Student is distilled end-to-end from the Teacher's representations
   using the three-component ``KnowledgeDistillationLoss``.
3. A cosine-annealing LR scheduler is applied to both optimisers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import SpecCompressNet, KnowledgeDistillationLoss, build_model, count_parameters
from preprocess import (
    frame_to_spectrum,
    simulate_compression,
    build_transforms,
    SPECTRUM_SIZE,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeepfakeSpectrumDataset(Dataset):
    """Paired dataset that yields (teacher_input, student_input, label) tuples.

    For each image the Teacher receives the raw RGB frame (spatial domain) and
    the Student receives the FFT spectrum of the *compressed* version of the
    same frame.  This separation mirrors the inference scenario where only
    compressed data is available to the deployed Student.

    Parameters
    ----------
    real_dir:
        Directory containing frames of *real* (label = 0) content.
    fake_dir:
        Directory containing frames of *deepfake* (label = 1) content.
    teacher_transform:
        Transform applied to the uncompressed Teacher input.
    student_transform:
        Transform applied to the compressed Student spectrum input.
    max_samples:
        Optional cap on the total number of samples (for quick debugging).
    """

    _VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(
        self,
        real_dir: str | Path,
        fake_dir: str | Path,
        teacher_transform=None,
        student_transform=None,
        max_samples: int | None = None,
    ) -> None:
        self.teacher_transform = teacher_transform
        self.student_transform = student_transform

        real_paths = self._collect_images(real_dir)
        fake_paths = self._collect_images(fake_dir)

        self.samples: list[tuple[Path, int]] = (
            [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
        )

        if max_samples is not None:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(self.samples), size=min(max_samples, len(self.samples)),
                             replace=False)
            self.samples = [self.samples[i] for i in idx]

        logger.info(
            "Dataset built: %d real, %d fake  (total %d)",
            len(real_paths), len(fake_paths), len(self.samples),
        )

    def _collect_images(self, directory: str | Path) -> list[Path]:
        d = Path(directory)
        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {d}")
        return sorted(
            p for p in d.iterdir()
            if p.suffix.lower() in self._VALID_EXTENSIONS
        )

    def __len__(self) -> int:  # noqa: D401
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load one sample and apply transforms.

        Returns
        -------
        teacher_tensor: ``(3, H, W)`` – uncompressed RGB frame for Teacher.
        student_tensor: ``(3, H, W)`` – compressed FFT spectrum for Student.
        label:          ``int``       – 0 (real) or 1 (fake).
        """
        path, label = self.samples[idx]
        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)  # (H, W, 3), uint8

        # Teacher input: raw RGB
        teacher_pil = Image.fromarray(img_np)
        teacher_tensor = (
            self.teacher_transform(teacher_pil)
            if self.teacher_transform else torch.tensor(img_np).permute(2, 0, 1).float() / 255.0
        )

        # Student input: FFT spectrum of compressed frame
        import cv2
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        compressed_bgr = simulate_compression(img_bgr)
        spectrum = frame_to_spectrum(compressed_bgr)  # (H, W, 3) float32 [0,1]
        spectrum_pil = Image.fromarray((spectrum * 255).astype(np.uint8))
        student_tensor = (
            self.student_transform(spectrum_pil)
            if self.student_transform else torch.tensor(spectrum).permute(2, 0, 1)
        )

        return teacher_tensor, student_tensor, label


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy for a batch.

    Parameters
    ----------
    logits: ``(B, C)``
    labels: ``(B,)``

    Returns
    -------
    float in [0, 1].
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: SpecCompressNet,
    loader: DataLoader,
    criterion: KnowledgeDistillationLoss,
    teacher_opt: torch.optim.Optimizer,
    student_opt: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
) -> dict[str, float]:
    """Run one full training epoch.

    Parameters
    ----------
    model:          The ``SpecCompressNet`` instance.
    loader:         Training ``DataLoader``.
    criterion:      ``KnowledgeDistillationLoss`` instance.
    teacher_opt:    Optimiser for the Teacher's unfrozen parameters.
    student_opt:    Optimiser for all Student parameters.
    device:         Target device.
    epoch:          Current epoch index (for logging only).
    log_interval:   Print a progress line every ``log_interval`` batches.

    Returns
    -------
    dict with keys ``"loss"``, ``"teacher_acc"``, ``"student_acc"``.
    """
    model.train()
    total_loss = 0.0
    teacher_correct = 0
    student_correct = 0
    n_samples = 0

    for batch_idx, (x_t, x_s, labels) in enumerate(loader):
        x_t = x_t.to(device)
        x_s = x_s.to(device)
        labels = labels.to(device)

        # Forward
        t_logits, t_features, s_logits, s_features = model(x_t, x_s)
        loss, components = criterion(t_logits, t_features, s_logits, s_features, labels)

        # Backward
        teacher_opt.zero_grad()
        student_opt.zero_grad()
        loss.backward()
        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(model.student.parameters(), max_norm=5.0)
        teacher_opt.step()
        student_opt.step()

        # Metrics
        bs = labels.size(0)
        total_loss += components["total"] * bs
        teacher_correct += (t_logits.argmax(1) == labels).sum().item()
        student_correct += (s_logits.argmax(1) == labels).sum().item()
        n_samples += bs

        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                "Epoch %3d | Batch %4d/%4d | Loss %.4f "
                "(task=%.4f feat=%.4f resp=%.4f) | "
                "T-Acc %.3f | S-Acc %.3f",
                epoch, batch_idx + 1, len(loader),
                components["total"], components["task"],
                components["feature"], components["response"],
                teacher_correct / n_samples,
                student_correct / n_samples,
            )

    return {
        "loss": total_loss / n_samples,
        "teacher_acc": teacher_correct / n_samples,
        "student_acc": student_correct / n_samples,
    }


@torch.no_grad()
def validate(
    model: SpecCompressNet,
    loader: DataLoader,
    criterion: KnowledgeDistillationLoss,
    device: torch.device,
) -> dict[str, float]:
    """Run a full validation pass.

    Parameters
    ----------
    model:     ``SpecCompressNet`` in eval mode.
    loader:    Validation ``DataLoader``.
    criterion: ``KnowledgeDistillationLoss`` instance.
    device:    Target device.

    Returns
    -------
    dict with keys ``"loss"``, ``"teacher_acc"``, ``"student_acc"``.
    """
    model.eval()
    total_loss = 0.0
    teacher_correct = 0
    student_correct = 0
    n_samples = 0

    for x_t, x_s, labels in loader:
        x_t = x_t.to(device)
        x_s = x_s.to(device)
        labels = labels.to(device)

        t_logits, t_features, s_logits, s_features = model(x_t, x_s)
        _, components = criterion(t_logits, t_features, s_logits, s_features, labels)

        bs = labels.size(0)
        total_loss += components["total"] * bs
        teacher_correct += (t_logits.argmax(1) == labels).sum().item()
        student_correct += (s_logits.argmax(1) == labels).sum().item()
        n_samples += bs

    return {
        "loss": total_loss / n_samples,
        "teacher_acc": teacher_correct / n_samples,
        "student_acc": student_correct / n_samples,
    }


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: SpecCompressNet,
    teacher_opt: torch.optim.Optimizer,
    student_opt: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    output_dir: str | Path,
    tag: str = "latest",
) -> Path:
    """Persist model weights and training state to a ``.pth`` file.

    Parameters
    ----------
    model:        ``SpecCompressNet`` instance.
    teacher_opt:  Teacher optimiser state.
    student_opt:  Student optimiser state.
    epoch:        Current epoch (0-indexed).
    metrics:      Dict of validation metrics to embed in the checkpoint.
    output_dir:   Directory where the file will be saved.
    tag:          File name suffix (e.g. ``"best"`` or ``"latest"``).

    Returns
    -------
    Path to the saved checkpoint file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"speccompressnet_{tag}_ep{epoch:03d}.pth"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "teacher_opt_state_dict": teacher_opt.state_dict(),
            "student_opt_state_dict": student_opt.state_dict(),
            "metrics": metrics,
        },
        ckpt_path,
    )
    logger.info("Checkpoint saved → %s", ckpt_path)
    return ckpt_path


def load_checkpoint(
    model: SpecCompressNet,
    checkpoint_path: str | Path,
    teacher_opt: torch.optim.Optimizer | None = None,
    student_opt: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> int:
    """Restore model (and optionally optimiser) state from a checkpoint.

    Parameters
    ----------
    model:            ``SpecCompressNet`` instance to restore into.
    checkpoint_path:  Path to a ``.pth`` file created by ``save_checkpoint``.
    teacher_opt:      If provided, its state is also restored.
    student_opt:      If provided, its state is also restored.
    device:           Device to map tensors to.

    Returns
    -------
    int
        The epoch at which the checkpoint was saved.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if teacher_opt and "teacher_opt_state_dict" in ckpt:
        teacher_opt.load_state_dict(ckpt["teacher_opt_state_dict"])
    if student_opt and "student_opt_state_dict" in ckpt:
        student_opt.load_state_dict(ckpt["student_opt_state_dict"])
    logger.info("Checkpoint loaded ← %s (epoch %d)", checkpoint_path, ckpt["epoch"])
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Full training pipeline.

    Parameters
    ----------
    args:
        Parsed command-line arguments (see ``build_arg_parser``).
    """
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- Transforms ----
    teacher_tf = build_transforms(compressed=False)
    student_tf = build_transforms(compressed=True)

    # ---- Dataset & splits ----
    full_dataset = DeepfakeSpectrumDataset(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        teacher_transform=teacher_tf,
        student_transform=student_tf,
        max_samples=args.max_samples,
    )

    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ---- Model ----
    model = build_model(device=device)
    logger.info("Teacher params: %s", count_parameters(model.teacher))
    logger.info("Student params: %s", count_parameters(model.student))

    # ---- Optimisers ----
    # Teacher: only fine-tune unfrozen parameters (layer4 + classifier)
    teacher_params = [p for p in model.teacher.parameters() if p.requires_grad]
    teacher_opt = torch.optim.AdamW(
        teacher_params, lr=args.lr * 0.1, weight_decay=1e-4
    )
    student_opt = torch.optim.AdamW(
        model.student.parameters(), lr=args.lr, weight_decay=1e-4
    )

    # ---- LR schedulers ----
    teacher_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        teacher_opt, T_max=args.epochs, eta_min=1e-6
    )
    student_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        student_opt, T_max=args.epochs, eta_min=1e-6
    )

    # ---- Loss ----
    criterion = KnowledgeDistillationLoss(
        temperature=args.kd_temperature,
        alpha=args.kd_alpha,
        beta=args.kd_beta,
    )

    # ---- Optionally resume ----
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            model, args.resume, teacher_opt, student_opt, device
        )

    # ---- Training loop ----
    history: list[dict] = []
    best_val_acc = 0.0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion,
            teacher_opt, student_opt,
            device, epoch + 1, args.log_interval,
        )
        val_metrics = validate(model, val_loader, criterion, device)

        teacher_sched.step()
        student_sched.step()

        elapsed = time.time() - epoch_start
        logger.info(
            "── Epoch %3d/%3d  (%5.1f s) ──  "
            "train_loss=%.4f  val_loss=%.4f  "
            "train_S-acc=%.3f  val_S-acc=%.3f",
            epoch + 1, args.epochs, elapsed,
            train_metrics["loss"], val_metrics["loss"],
            train_metrics["student_acc"], val_metrics["student_acc"],
        )

        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        # Save latest checkpoint
        save_checkpoint(
            model, teacher_opt, student_opt,
            epoch + 1, val_metrics,
            args.output_dir, tag="latest",
        )

        # Save best checkpoint (by Student validation accuracy)
        if val_metrics["student_acc"] > best_val_acc:
            best_val_acc = val_metrics["student_acc"]
            save_checkpoint(
                model, teacher_opt, student_opt,
                epoch + 1, val_metrics,
                args.output_dir, tag="best",
            )
            logger.info("★ New best val Student-acc: %.4f", best_val_acc)

    # ---- Persist training history ----
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, "w") as fh:
        json.dump(history, fh, indent=2)
    logger.info("Training history saved → %s", history_path)
    logger.info("Training complete. Best val Student-acc: %.4f", best_val_acc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Train SpecCompress-Net with Knowledge Distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--real_dir", required=True,
                   help="Directory with real video frames.")
    p.add_argument("--fake_dir", required=True,
                   help="Directory with deepfake video frames.")
    p.add_argument("--val_split", type=float, default=0.15,
                   help="Fraction of data used for validation.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap total samples for quick debugging.")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Base learning rate for the Student.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=20,
                   help="Print log every N batches.")

    # KD hyper-parameters
    p.add_argument("--kd_temperature", type=float, default=4.0)
    p.add_argument("--kd_alpha", type=float, default=0.5,
                   help="Weight of distillation loss (0=task only).")
    p.add_argument("--kd_beta", type=float, default=0.3,
                   help="Feature vs response distillation split.")

    # I/O
    p.add_argument("--output_dir", default="checkpoints",
                   help="Directory for checkpoint files.")
    p.add_argument("--resume", default=None,
                   help="Path to a .pth checkpoint to resume from.")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    train(cli_args)
