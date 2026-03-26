"""
visualize.py
============
Visualisation utilities for SpecCompress-Net.

Generates side-by-side comparison figures that illustrate the difference
between real and deepfake frames in both the spatial domain (RGB) and the
frequency domain (FFT magnitude spectrum), with and without compression
simulation.

Typical usage
-------------
.. code-block:: bash

    python visualize.py \\
        --real   data/real/frame_0001.jpg \\
        --fake   data/fake/frame_0001.jpg \\
        --output figures/comparison.png

Alternatively, import and call directly from a notebook::

    from visualize import plot_comparison, plot_spectrum_grid
    plot_comparison("real.jpg", "fake.jpg")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from preprocess import (
    frame_to_spectrum,
    simulate_compression,
    SPECTRUM_SIZE,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

_CMAP_SPECTRUM = "inferno"     # Perceptually uniform, highlights energy peaks
_CMAP_DIFF = "RdBu_r"         # Diverging map for difference images


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_bgr(path: str | Path) -> np.ndarray:
    """Load an image as a BGR uint8 numpy array.

    Parameters
    ----------
    path:
        File path to any image format supported by OpenCV.

    Returns
    -------
    np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or cannot be decoded.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return bgr


def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to RGB for Matplotlib display."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _resize(frame_bgr: np.ndarray, size: int = SPECTRUM_SIZE) -> np.ndarray:
    """Resize a frame to ``size × size`` pixels."""
    return cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)


def _spectrum_single_channel(frame_bgr: np.ndarray) -> np.ndarray:
    """Return the log-magnitude spectrum as a single float32 channel.

    Parameters
    ----------
    frame_bgr: ``(H, W, 3)`` uint8.

    Returns
    -------
    np.ndarray
        Normalised log-magnitude spectrum, shape ``(H, W)``, float32 in [0, 1].
    """
    spec_3ch = frame_to_spectrum(frame_bgr)
    return spec_3ch[:, :, 0]   # All three channels are identical


# ---------------------------------------------------------------------------
# Main comparison figure
# ---------------------------------------------------------------------------

def plot_comparison(
    real_path: str | Path,
    fake_path: str | Path,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (18, 10),
) -> plt.Figure:
    """Generate a 2×5 comparison grid for one real and one deepfake frame.

    Grid layout (rows = Real / Fake, columns = 5 stages)
    ------------------------------------------------------
    Col 0: Original RGB frame
    Col 1: Compressed RGB frame
    Col 2: FFT spectrum of original
    Col 3: FFT spectrum of compressed
    Col 4: Difference spectrum  |spec_orig − spec_compressed|

    Parameters
    ----------
    real_path:
        Path to a real video frame image.
    fake_path:
        Path to a deepfake video frame image.
    output_path:
        If provided, saves the figure to this path (PNG / PDF / SVG).
    figsize:
        Matplotlib figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    real_bgr = _resize(_load_bgr(real_path))
    fake_bgr = _resize(_load_bgr(fake_path))

    real_comp = simulate_compression(real_bgr)
    fake_comp = simulate_compression(fake_bgr)

    real_spec_orig = _spectrum_single_channel(real_bgr)
    real_spec_comp = _spectrum_single_channel(real_comp)
    fake_spec_orig = _spectrum_single_channel(fake_bgr)
    fake_spec_comp = _spectrum_single_channel(fake_comp)

    real_diff = np.abs(real_spec_orig - real_spec_comp)
    fake_diff = np.abs(fake_spec_orig - fake_spec_comp)

    col_titles = [
        "Original RGB",
        "Compressed RGB",
        "FFT Spectrum\n(Original)",
        "FFT Spectrum\n(Compressed)",
        "Difference\n|Orig − Comp|",
    ]
    row_labels = ["Real", "Deepfake"]

    fig, axes = plt.subplots(2, 5, figsize=figsize)
    fig.suptitle(
        "SpecCompress-Net – Spatial vs Frequency Domain Analysis",
        fontsize=14, fontweight="bold", y=1.02,
    )

    data_grid = [
        # real row
        [
            _bgr_to_rgb(real_bgr),
            _bgr_to_rgb(real_comp),
            real_spec_orig,
            real_spec_comp,
            real_diff,
        ],
        # fake row
        [
            _bgr_to_rgb(fake_bgr),
            _bgr_to_rgb(fake_comp),
            fake_spec_orig,
            fake_spec_comp,
            fake_diff,
        ],
    ]

    for row_idx, (row_data, row_label) in enumerate(zip(data_grid, row_labels)):
        for col_idx, img_data in enumerate(row_data):
            ax = axes[row_idx, col_idx]
            is_spectrum = col_idx >= 2

            if is_spectrum:
                cmap = _CMAP_DIFF if col_idx == 4 else _CMAP_SPECTRUM
                im = ax.imshow(img_data, cmap=cmap, vmin=0.0, vmax=1.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(img_data)

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(
                    row_label,
                    fontsize=12, fontweight="bold",
                    color="steelblue" if row_label == "Real" else "crimson",
                )
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved → {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Spectrum grid for a single sample
# ---------------------------------------------------------------------------

def plot_spectrum_grid(
    frame_bgr: np.ndarray,
    label: str = "Unknown",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Display a detailed four-panel analysis for a single frame.

    Panels
    ------
    1. Original RGB frame.
    2. Compressed RGB frame.
    3. Log-magnitude FFT spectrum (original) on an *inferno* colour map with
       annotated peak regions.
    4. Radially averaged power spectrum – shows how energy is distributed
       across spatial frequency bands.

    Parameters
    ----------
    frame_bgr:
        Input frame, shape ``(H, W, 3)``, dtype ``uint8``.
    label:
        Class label used in the figure title (``"Real"`` or ``"Deepfake"``).
    output_path:
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    frame_bgr = _resize(frame_bgr)
    frame_comp = simulate_compression(frame_bgr)

    spec_orig = _spectrum_single_channel(frame_bgr)
    radial_profile = _radial_power_spectrum(frame_bgr)

    colour = "steelblue" if label.lower() == "real" else "crimson"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Frequency-Domain Analysis  |  Sample: {label}",
        fontsize=13, fontweight="bold", color=colour,
    )

    # Panel 1 – original RGB
    axes[0].imshow(_bgr_to_rgb(frame_bgr))
    axes[0].set_title("Original Frame", fontsize=10)

    # Panel 2 – compressed RGB
    axes[1].imshow(_bgr_to_rgb(frame_comp))
    axes[1].set_title("Compressed Frame", fontsize=10)

    # Panel 3 – FFT spectrum
    im = axes[2].imshow(spec_orig, cmap=_CMAP_SPECTRUM, vmin=0, vmax=1)
    axes[2].set_title("Log FFT Spectrum", fontsize=10)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    _annotate_quadrants(axes[2], spec_orig.shape[0])

    # Panel 4 – radial power curve
    axes[3].plot(radial_profile, color=colour, linewidth=1.5)
    axes[3].fill_between(
        range(len(radial_profile)), radial_profile, alpha=0.15, color=colour
    )
    axes[3].set_title("Radial Power Spectrum", fontsize=10)
    axes[3].set_xlabel("Spatial Frequency (bin)")
    axes[3].set_ylabel("Normalised Power")
    axes[3].set_xlim(0, len(radial_profile) - 1)

    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved → {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Utility: radial power spectrum
# ---------------------------------------------------------------------------

def _radial_power_spectrum(
    frame_bgr: np.ndarray,
    n_bins: int = 100,
) -> np.ndarray:
    """Compute the radially averaged power spectrum of a frame.

    Converts the 2-D FFT into a 1-D profile by averaging the power within
    concentric annuli centred on the DC component.  This summarises how
    much energy the image has at each spatial frequency scale.

    Parameters
    ----------
    frame_bgr:
        Input frame, shape ``(H, W, 3)``.
    n_bins:
        Number of radial frequency bins.

    Returns
    -------
    np.ndarray
        Normalised 1-D power profile, shape ``(n_bins,)``, dtype float64.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    fft = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(fft) ** 2

    # Build a radial distance map from the centre
    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.indices((H, W))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    r_max = r.max()

    bins = np.linspace(0, r_max, n_bins + 1)
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.sum() > 0:
            profile[i] = power[mask].mean()

    # Normalise
    if profile.max() > 0:
        profile /= profile.max()
    return profile


def _annotate_quadrants(ax: plt.Axes, size: int) -> None:
    """Draw dashed lines dividing the FFT spectrum into four quadrants."""
    mid = size // 2
    ax.axhline(mid, color="white", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(mid, color="white", linewidth=0.6, linestyle="--", alpha=0.5)
    for label, (y, x) in {
        "LF": (mid // 2, mid // 2),
        "HF": (mid // 2, mid + mid // 2),
    }.items():
        ax.text(
            x, y, label, color="white", fontsize=7, ha="center", va="center",
            alpha=0.8,
        )


# ---------------------------------------------------------------------------
# Batch comparison utility
# ---------------------------------------------------------------------------

def compare_batch(
    real_paths: list[str | Path],
    fake_paths: list[str | Path],
    output_dir: str | Path = "figures",
) -> None:
    """Generate per-pair comparison figures for lists of real/fake paths.

    Parameters
    ----------
    real_paths:
        Ordered list of real frame paths.
    fake_paths:
        Ordered list of deepfake frame paths.  Must match length of
        ``real_paths``.
    output_dir:
        Directory where output PNG files are saved.
    """
    if len(real_paths) != len(fake_paths):
        raise ValueError("real_paths and fake_paths must have the same length.")

    output_dir = Path(output_dir)
    for idx, (rp, fp) in enumerate(zip(real_paths, fake_paths)):
        out = output_dir / f"comparison_{idx:04d}.png"
        plot_comparison(rp, fp, output_path=out)
        plt.close("all")
    print(f"Saved {len(real_paths)} comparison figures to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualise frame vs FFT spectrum for SpecCompress-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--real", required=True, help="Path to a real frame image.")
    p.add_argument("--fake", required=True, help="Path to a deepfake frame image.")
    p.add_argument("--output", default="figures/comparison.png",
                   help="Path to save the output figure.")
    p.add_argument("--no_show", action="store_true",
                   help="Do not display the figure interactively.")
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    fig = plot_comparison(args.real, args.fake, output_path=args.output)
    if not args.no_show:
        plt.show()
