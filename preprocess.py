"""
preprocess.py
=============
Preprocessing utilities for SpecCompress-Net.

Converts raw video frames into 2-D Fourier magnitude spectra that expose
compression-induced high-frequency artefacts, and simulates heavy video
compression (high CRF) through Gaussian blur and JPEG quantisation.

Typical usage
-------------
>>> from preprocess import frame_to_spectrum, simulate_compression, build_transforms
>>> spectrum = frame_to_spectrum(frame_bgr)          # numpy → numpy
>>> compressed = simulate_compression(frame_bgr)      # numpy → numpy
>>> tf = build_transforms(compressed=True)            # → torchvision.transforms
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

SPECTRUM_SIZE: int = 224          # Spatial size fed to the model (pixels)
LOG_EPSILON: float = 1e-8         # Guards against log(0)
JPEG_QUALITY: int = 30            # Low quality mimics high-CRF compression
GAUSSIAN_KERNEL: int = 5          # Blur kernel size for compression simulation
GAUSSIAN_SIGMA: float = 1.4       # Blur sigma for compression simulation


# ---------------------------------------------------------------------------
# Core preprocessing helpers
# ---------------------------------------------------------------------------

def frame_to_spectrum(
    frame_bgr: np.ndarray,
    output_size: int = SPECTRUM_SIZE,
) -> np.ndarray:
    """Convert a BGR video frame to a 2-D log-scaled FFT magnitude spectrum.

    Processing pipeline
    -------------------
    1. Convert BGR → grayscale.
    2. Resize to ``output_size × output_size``.
    3. Compute 2-D DFT and shift the zero-frequency component to the centre.
    4. Compute the magnitude spectrum.
    5. Apply log scaling: ``log(1 + magnitude)``.
    6. Min-max normalise to ``[0, 1]``.
    7. Stack the single-channel result into 3 channels so it is compatible
       with standard CNN input layers that expect RGB tensors.

    Parameters
    ----------
    frame_bgr:
        A video frame in BGR colour order as returned by ``cv2.imread`` /
        ``cv2.VideoCapture.read``, shape ``(H, W, 3)``, dtype ``uint8``.
    output_size:
        Side length (pixels) of the square output spectrum image.

    Returns
    -------
    np.ndarray
        Normalised log-magnitude spectrum, shape ``(output_size, output_size, 3)``,
        dtype ``float32``, values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``frame_bgr`` is not a 3-channel image.

    Examples
    --------
    >>> import cv2
    >>> frame = cv2.imread("sample.jpg")
    >>> spec = frame_to_spectrum(frame)
    >>> spec.shape
    (224, 224, 3)
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel BGR frame, got shape {frame_bgr.shape}."
        )

    # Step 1 – grayscale
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Step 2 – resize
    gray = cv2.resize(gray, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # Step 3 – 2-D FFT, centre shift
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)

    # Step 4 – magnitude
    magnitude = np.abs(fft_shifted)

    # Step 5 – log scaling
    log_magnitude = np.log(1.0 + magnitude + LOG_EPSILON)

    # Step 6 – min-max normalisation
    lo, hi = log_magnitude.min(), log_magnitude.max()
    if hi - lo > LOG_EPSILON:
        log_magnitude = (log_magnitude - lo) / (hi - lo)
    else:
        log_magnitude = np.zeros_like(log_magnitude)

    # Step 7 – replicate to 3 channels for CNN compatibility
    spectrum_3ch = np.stack([log_magnitude] * 3, axis=-1).astype(np.float32)
    return spectrum_3ch


def simulate_compression(
    frame_bgr: np.ndarray,
    jpeg_quality: int = JPEG_QUALITY,
    kernel_size: int = GAUSSIAN_KERNEL,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Simulate the visual degradation produced by high-CRF video compression.

    Two degradation stages are applied sequentially:

    1. **Gaussian blur** – models the low-pass filtering effect of intra-frame
       spatial compression.
    2. **JPEG re-encoding at low quality** – models the quantisation artefacts
       introduced by inter-frame codecs (H.264/H.265 at high CRF values).

    Parameters
    ----------
    frame_bgr:
        Input frame, shape ``(H, W, 3)``, dtype ``uint8``.
    jpeg_quality:
        JPEG quality factor ``[1, 95]``; lower values = heavier compression.
    kernel_size:
        Side length of the Gaussian kernel (must be odd and positive).
    sigma:
        Standard deviation for the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Compressed frame, same shape and dtype as the input.

    Examples
    --------
    >>> compressed = simulate_compression(frame, jpeg_quality=20)
    """
    # Stage 1 – spatial blur
    if kernel_size % 2 == 0:
        kernel_size += 1  # cv2 requires an odd kernel size
    blurred = cv2.GaussianBlur(frame_bgr, (kernel_size, kernel_size), sigma)

    # Stage 2 – JPEG quantisation artefacts
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(95, jpeg_quality))]
    _, encoded = cv2.imencode(".jpg", blurred, encode_params)
    compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return compressed


def frame_to_rgb_tensor(
    frame_bgr: np.ndarray,
    output_size: int = SPECTRUM_SIZE,
) -> np.ndarray:
    """Resize a BGR frame and convert it to a normalised RGB float32 array.

    Used to feed *raw pixel* data to the Teacher model, which operates in the
    spatial domain.

    Parameters
    ----------
    frame_bgr:
        Input frame in BGR colour order, shape ``(H, W, 3)``, dtype ``uint8``.
    output_size:
        Target side length in pixels.

    Returns
    -------
    np.ndarray
        RGB image, shape ``(output_size, output_size, 3)``, dtype ``float32``,
        values in ``[0, 1]``.
    """
    resized = cv2.resize(frame_bgr, (output_size, output_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def build_transforms(compressed: bool = False) -> transforms.Compose:
    """Return a ``torchvision.transforms.Compose`` pipeline for model input.

    The pipeline converts a ``PIL.Image`` or ``numpy.ndarray`` (H×W×3, float32,
    [0,1]) to a normalised ``torch.Tensor`` ready for the CNN backbone.

    ImageNet statistics are used for normalisation because both the Teacher
    and Student backbones are initialised from ImageNet pre-trained weights.

    Parameters
    ----------
    compressed:
        If ``True``, include a light random-augmentation stage (horizontal flip
        and colour jitter) to help the Student model generalise across different
        compression levels.

    Returns
    -------
    torchvision.transforms.Compose
        A composed transform pipeline.

    Examples
    --------
    >>> tf = build_transforms(compressed=True)
    >>> tensor = tf(pil_image)          # shape: (3, 224, 224)
    """
    base = [
        transforms.Resize((SPECTRUM_SIZE, SPECTRUM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    if compressed:
        augment = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        # Insert augmentations before ToTensor
        pipeline = augment + base
    else:
        pipeline = base

    return transforms.Compose(pipeline)


# ---------------------------------------------------------------------------
# Dataset-level helper
# ---------------------------------------------------------------------------

def preprocess_frame_pair(
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return both the raw spectrum and the compressed spectrum for one frame.

    Convenience wrapper that bundles ``frame_to_spectrum`` and
    ``simulate_compression`` + ``frame_to_spectrum`` into a single call,
    producing the (Teacher input, Student input) pair used during training.

    Parameters
    ----------
    frame_bgr:
        Raw video frame in BGR colour order, ``uint8``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(raw_spectrum, compressed_spectrum)`` – both are float32 arrays of
        shape ``(SPECTRUM_SIZE, SPECTRUM_SIZE, 3)`` with values in ``[0, 1]``.
    """
    raw_spectrum = frame_to_spectrum(frame_bgr)
    compressed_frame = simulate_compression(frame_bgr)
    compressed_spectrum = frame_to_spectrum(compressed_frame)
    return raw_spectrum, compressed_spectrum
