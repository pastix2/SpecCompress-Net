"""
utils.py
========
General-purpose utilities for the SpecCompress-Net pipeline.

Contents
--------
* ``extract_frames``         – Extract frames from a video file.
* ``evaluate_model``         – Full evaluation loop with AUC / F1 reporting.
* ``plot_training_history``  – Parse and visualise a ``training_history.json``.
* ``seed_everything``        – Reproducibility helper.
* ``get_device``             – Select CUDA / MPS / CPU automatically.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Optional heavy imports – only imported when the relevant function is called
# to keep the module lightweight in minimal environments.
try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training.

    Sets seeds for Python's ``random``, ``numpy``, ``torch`` (CPU & CUDA),
    and enables deterministic CuDNN algorithms.

    Parameters
    ----------
    seed:
        Integer seed value.

    Examples
    --------
    >>> seed_everything(2026)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available ``torch.device``.

    Priority: CUDA GPU → Apple MPS → CPU.

    Parameters
    ----------
    prefer_gpu:
        If ``False``, always return CPU regardless of hardware availability.

    Returns
    -------
    torch.device

    Examples
    --------
    >>> device = get_device()
    >>> print(device)
    cuda
    """
    if not prefer_gpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: int = 1,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
) -> list[Path]:
    """Extract individual frames from a video file and save them as JPEG images.

    Parameters
    ----------
    video_path:
        Path to the source video file (MP4, AVI, MOV, etc.).
    output_dir:
        Directory where extracted frames will be saved.  Created automatically.
    fps:
        Number of frames to extract per second of video.
        Set to ``0`` to extract every frame.
    max_frames:
        Maximum number of frames to save.  Extraction stops once this limit is
        reached (useful for building balanced datasets).
    resize:
        Optional ``(width, height)`` tuple.  If provided every frame is
        resized before saving.

    Returns
    -------
    list[Path]
        Sorted list of paths to the saved frame images.

    Raises
    ------
    FileNotFoundError
        If ``video_path`` cannot be opened by OpenCV.

    Examples
    --------
    >>> paths = extract_frames("video.mp4", "frames/real/", fps=2, max_frames=500)
    >>> len(paths)
    500
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0  # fallback

    frame_interval = max(1, round(video_fps / fps)) if fps > 0 else 1

    saved_paths: list[Path] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            stem = f"frame_{frame_idx:07d}.jpg"
            out_path = output_dir / stem
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(out_path)

            if max_frames is not None and len(saved_paths) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return sorted(saved_paths)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    student_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    """Evaluate the Student model on a dataset and return full metrics.

    Computes accuracy, AUC-ROC (requires ``scikit-learn``), F1-score, and
    prints a classification report.

    Parameters
    ----------
    student_model:
        The trained Student (or any ``nn.Module`` that accepts a single tensor
        input and returns ``(logits, features)``).
    data_loader:
        ``DataLoader`` whose items are ``(_, student_tensor, label)`` tuples.
        The first element (Teacher input) is ignored.
    device:
        Target device.
    class_names:
        Human-readable class names for the report
        (e.g. ``["Real", "Deepfake"]``).

    Returns
    -------
    dict
        Keys: ``"accuracy"``, ``"auc"`` (if sklearn available),
        ``"f1_macro"``, ``"report"``.
    """
    if class_names is None:
        class_names = ["Real", "Deepfake"]

    student_model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for batch in data_loader:
        # Support both (teacher, student, label) and (student, label) formats
        if len(batch) == 3:
            _, x_s, labels = batch
        else:
            x_s, labels = batch

        x_s = x_s.to(device)
        logits, _ = student_model(x_s)
        probs = torch.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())   # prob of deepfake
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    results: dict = {"accuracy": float(accuracy)}

    if _SKLEARN_AVAILABLE:
        auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds, average="macro")
        report = classification_report(all_labels, all_preds,
                                        target_names=class_names)
        cm = confusion_matrix(all_labels, all_preds)
        results.update({
            "auc": float(auc),
            "f1_macro": float(f1),
            "report": report,
            "confusion_matrix": cm.tolist(),
        })
        print(f"\nAccuracy : {accuracy:.4f}")
        print(f"AUC-ROC  : {auc:.4f}")
        print(f"F1-Macro : {f1:.4f}")
        print("\nClassification Report:")
        print(report)
    else:
        print(
            "scikit-learn not installed — skipping AUC / F1 metrics. "
            "Install with: pip install scikit-learn"
        )

    return results


# ---------------------------------------------------------------------------
# Training curve visualisation
# ---------------------------------------------------------------------------

def plot_training_history(
    history_path: str | Path,
    output_path: str | Path | None = None,
) -> "plt.Figure":  # noqa: F821
    """Read ``training_history.json`` and plot loss / accuracy curves.

    Parameters
    ----------
    history_path:
        Path to the JSON file written by ``train.py``.
    output_path:
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    """
    if not _MPL_AVAILABLE:
        raise ImportError("Matplotlib is required. Install with: pip install matplotlib")

    with open(history_path) as fh:
        history = json.load(fh)

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train"]["loss"] for r in history]
    val_loss = [r["val"]["loss"] for r in history]
    train_acc = [r["train"]["student_acc"] for r in history]
    val_acc = [r["val"]["student_acc"] for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SpecCompress-Net – Training History", fontsize=13, fontweight="bold")

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="steelblue", linewidth=1.5)
    ax1.plot(epochs, val_loss, label="Val Loss", color="crimson",
             linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("KD Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in train_acc],
             label="Train Acc (Student)", color="steelblue", linewidth=1.5)
    ax2.plot(epochs, [a * 100 for a in val_acc],
             label="Val Acc (Student)", color="crimson",
             linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Student Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Training history plot saved → {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_single_frame(
    frame_bgr: np.ndarray,
    student_model: nn.Module,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Run inference on a single frame and return label probabilities.

    Parameters
    ----------
    frame_bgr:
        Raw BGR video frame, shape ``(H, W, 3)``, dtype ``uint8``.
    student_model:
        Trained Student model in eval mode.
    device:
        Device to run inference on.
    class_names:
        Human-readable class labels (default ``["Real", "Deepfake"]``).

    Returns
    -------
    dict
        Mapping ``{class_name: probability}`` plus ``"predicted_class"``.

    Examples
    --------
    >>> result = predict_single_frame(frame, student, device)
    >>> print(result)
    {'Real': 0.07, 'Deepfake': 0.93, 'predicted_class': 'Deepfake'}
    """
    from preprocess import frame_to_spectrum, simulate_compression, build_transforms
    from PIL import Image

    if class_names is None:
        class_names = ["Real", "Deepfake"]

    student_model.eval()
    tf = build_transforms(compressed=True)

    compressed = simulate_compression(frame_bgr)
    spectrum = frame_to_spectrum(compressed)  # (H, W, 3) float32 [0, 1]
    spectrum_pil = Image.fromarray((spectrum * 255).astype(np.uint8))
    tensor = tf(spectrum_pil).unsqueeze(0).to(device)   # (1, 3, H, W)

    with torch.no_grad():
        logits, _ = student_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    result = {name: float(p) for name, p in zip(class_names, probs)}
    result["predicted_class"] = class_names[int(probs.argmax())]
    return result
