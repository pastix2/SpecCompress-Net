"""
model.py
========
SpecCompress-Net architecture: Teacher-Student Knowledge Distillation for
deepfake detection under heavy video compression.

Architecture overview
---------------------

Teacher (``TeacherModel``)
    ResNet-50 backbone pre-trained on ImageNet.  Processes *uncompressed* RGB
    frames or their FFT spectra.  All convolutional layers except the final
    block (``layer4``) are frozen during training so that fine-tuning is fast
    and stable.

Student (``StudentModel``)
    Lightweight custom CNN with four convolutional stages.  Processes
    *compressed* FFT magnitude spectra.  Designed to be 4-5× faster and
    ~10× smaller than the Teacher.

SpecCompressNet (``SpecCompressNet``)
    Wrapper that holds both sub-models and exposes the joint forward pass used
    during Knowledge-Distillation training.  At inference only the Student is
    required.

Knowledge Distillation Loss (``KnowledgeDistillationLoss``)
    Combines:
    * ``CrossEntropyLoss`` on the Student's logits (task loss).
    * ``MSELoss`` between Teacher and Student intermediate feature maps
      (feature-level distillation).
    * ``KLDivLoss`` between softened Teacher and Student output distributions
      (response-level distillation / *Hinton* loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 2          # binary: real (0) vs deepfake (1)
FEATURE_DIM: int = 512        # Student's bottleneck feature dimension
KD_TEMPERATURE: float = 4.0   # Softening temperature for response distillation
ALPHA: float = 0.5            # Weight of the distillation loss vs task loss
BETA: float = 0.3             # Weight of the feature-distillation component


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class _ConvBnRelu(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU block with ``same``-style padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _DepthwiseSeparable(nn.Sequential):
    """Depthwise-separable convolution block used in the Student encoder."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# Teacher model
# ---------------------------------------------------------------------------

class TeacherModel(nn.Module):
    """ResNet-50 Teacher that processes uncompressed data.

    The backbone is initialised from ImageNet pre-trained weights.  Layers
    ``conv1``, ``bn1``, ``layer1``, ``layer2``, and ``layer3`` are frozen.
    Only ``layer4`` and the classification head are updated during training.

    An additional projection head maps ``layer3`` features to ``FEATURE_DIM``
    for alignment with the Student's bottleneck, enabling feature-level
    Knowledge Distillation.

    Parameters
    ----------
    num_classes:
        Number of output classes (default ``2`` for binary deepfake detection).
    pretrained:
        Whether to initialise the backbone from ImageNet weights.

    Attributes
    ----------
    backbone:
        The ResNet-50 model with its original fully-connected head replaced.
    feature_projector:
        1×1 convolution that projects ``layer3`` output (1024 channels) to
        ``FEATURE_DIM`` channels for distillation alignment.
    classifier:
        Two-layer MLP classification head.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Extract feature extractor components
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3   # output: (B, 1024, 14, 14)
        self.layer4 = backbone.layer4   # output: (B, 2048, 7,  7)
        self.avgpool = backbone.avgpool

        # Freeze everything up to (and including) layer3
        for module in [self.stem, self.layer1, self.layer2, self.layer3]:
            for param in module.parameters():
                param.requires_grad = False

        # Projection head for KD feature alignment (layer3 → FEATURE_DIM)
        self.feature_projector = nn.Sequential(
            nn.Conv2d(1024, FEATURE_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(FEATURE_DIM),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor, shape ``(B, 3, H, W)``.

        Returns
        -------
        logits:
            Class logits, shape ``(B, num_classes)``.
        distill_features:
            Projected ``layer3`` feature vector, shape ``(B, FEATURE_DIM)``,
            used as the Teacher's "soft target" in feature distillation.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        layer3_out = self.layer3(x)
        x = self.layer4(layer3_out)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # Distillation features from the intermediate representation
        distill_features = self.feature_projector(layer3_out)
        distill_features = torch.flatten(distill_features, 1)  # (B, FEATURE_DIM)

        return logits, distill_features


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------

class StudentModel(nn.Module):
    """Lightweight Student CNN that processes compressed FFT spectra.

    Architecture summary (for 224×224 input)
    -----------------------------------------
    Stage 1  – 2× ConvBnRelu  (3   → 32 ch, 112×112)
    Stage 2  – 2× DepthwiseSep (32  → 64 ch,  56× 56)
    Stage 3  – 3× DepthwiseSep (64  → 128 ch, 28× 28)
    Stage 4  – 2× DepthwiseSep (128 → 256 ch, 14× 14)
    Global Average Pool → (B, 256)
    Projection MLP     → (B, FEATURE_DIM)
    Classifier         → (B, num_classes)

    Total parameters: ~1.2 M (vs ~25 M for ResNet-50).

    Parameters
    ----------
    num_classes:
        Number of output classes.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        # --- Encoder ---
        self.stage1 = nn.Sequential(
            _ConvBnRelu(3,  32, stride=2),
            _ConvBnRelu(32, 32),
        )
        self.stage2 = nn.Sequential(
            _DepthwiseSeparable(32, 64, stride=2),
            _DepthwiseSeparable(64, 64),
        )
        self.stage3 = nn.Sequential(
            _DepthwiseSeparable(64,  128, stride=2),
            _DepthwiseSeparable(128, 128),
            _DepthwiseSeparable(128, 128),
        )
        self.stage4 = nn.Sequential(
            _DepthwiseSeparable(128, 256, stride=2),
            _DepthwiseSeparable(256, 256),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection to FEATURE_DIM for KD alignment
        self.feature_projector = nn.Sequential(
            nn.Linear(256, FEATURE_DIM),
            nn.ReLU(inplace=True),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(FEATURE_DIM, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
            Compressed FFT spectrum tensor, shape ``(B, 3, H, W)``.

        Returns
        -------
        logits:
            Class logits, shape ``(B, num_classes)``.
        distill_features:
            Projected bottleneck vector, shape ``(B, FEATURE_DIM)``,
            aligned with the Teacher's ``distill_features`` for KD.
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)           # (B, 256)

        distill_features = self.feature_projector(x)   # (B, FEATURE_DIM)
        logits = self.classifier(distill_features)

        return logits, distill_features


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------

class SpecCompressNet(nn.Module):
    """Unified wrapper exposing the Teacher-Student pair for KD training.

    During **training** both models are active:
    - The Teacher receives uncompressed data and produces soft labels +
      intermediate features.
    - The Student receives compressed FFT spectra and is distilled towards the
      Teacher via the ``KnowledgeDistillationLoss``.

    During **inference** only the Student is needed::

        model.eval()
        logits, _ = model.student(compressed_spectrum)
        pred = logits.argmax(dim=1)

    Parameters
    ----------
    num_classes:
        Number of output classes.
    pretrained_teacher:
        Whether to initialise the Teacher backbone from ImageNet weights.

    Attributes
    ----------
    teacher: TeacherModel
    student: StudentModel
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained_teacher: bool = True,
    ) -> None:
        super().__init__()
        self.teacher = TeacherModel(num_classes=num_classes,
                                    pretrained=pretrained_teacher)
        self.student = StudentModel(num_classes=num_classes)

    def forward(
        self,
        x_teacher: torch.Tensor,
        x_student: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint forward pass used during KD training.

        Parameters
        ----------
        x_teacher:
            Uncompressed input for the Teacher, shape ``(B, 3, H, W)``.
        x_student:
            Compressed FFT spectrum for the Student, shape ``(B, 3, H, W)``.

        Returns
        -------
        teacher_logits:    shape ``(B, num_classes)``
        teacher_features:  shape ``(B, FEATURE_DIM)``
        student_logits:    shape ``(B, num_classes)``
        student_features:  shape ``(B, FEATURE_DIM)``
        """
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher(x_teacher)

        student_logits, student_features = self.student(x_student)
        return teacher_logits, teacher_features, student_logits, student_features

    def student_only(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only forward pass through the Student.

        Parameters
        ----------
        x:
            Compressed FFT spectrum tensor, shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Predicted class probabilities, shape ``(B, num_classes)``.
        """
        logits, _ = self.student(x)
        return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Knowledge Distillation loss
# ---------------------------------------------------------------------------

class KnowledgeDistillationLoss(nn.Module):
    """Three-component Knowledge Distillation loss for SpecCompress-Net.

    Loss breakdown
    --------------
    ``L_task``
        Standard cross-entropy on the **Student's** logits vs. ground-truth
        labels.  Ensures the Student learns the actual classification task.

    ``L_response`` (Hinton KD loss)
        KL divergence between the temperature-softened Teacher and Student
        output distributions.  Transfers *inter-class relationship* knowledge.

        .. math::
            L_{resp} = T^{2} \\cdot \\text{KLDiv}(
                \\sigma(z_s / T),\\, \\sigma(z_t / T)
            )

    ``L_feature``
        Mean-squared error between the Teacher's and Student's projected
        intermediate feature vectors.  Transfers *representational* knowledge.

    The combined loss is:

    .. math::
        L = (1 - \\alpha) \\cdot L_{task}
          + \\alpha \\cdot \\beta  \\cdot L_{feature}
          + \\alpha \\cdot (1 - \\beta) \\cdot L_{response}

    Parameters
    ----------
    temperature:
        Softening temperature ``T`` for the response distillation.  Higher
        values produce softer probability distributions.
    alpha:
        Overall weight of the distillation terms relative to the task loss.
    beta:
        Split between feature-level (``beta``) and response-level
        (``1 - beta``) distillation.

    Examples
    --------
    >>> criterion = KnowledgeDistillationLoss()
    >>> loss = criterion(
    ...     teacher_logits, teacher_features,
    ...     student_logits, student_features,
    ...     labels,
    ... )
    """

    def __init__(
        self,
        temperature: float = KD_TEMPERATURE,
        alpha: float = ALPHA,
        beta: float = BETA,
    ) -> None:
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.beta = beta

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        teacher_logits: torch.Tensor,
        teacher_features: torch.Tensor,
        student_logits: torch.Tensor,
        student_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined Knowledge Distillation loss.

        Parameters
        ----------
        teacher_logits:    ``(B, num_classes)`` – Teacher's output logits.
        teacher_features:  ``(B, FEATURE_DIM)``  – Teacher's intermediate features.
        student_logits:    ``(B, num_classes)`` – Student's output logits.
        student_features:  ``(B, FEATURE_DIM)``  – Student's intermediate features.
        labels:            ``(B,)`` int64         – Ground-truth class indices.

        Returns
        -------
        total_loss:
            Scalar ``torch.Tensor`` ready for ``.backward()``.
        loss_components:
            Dict with float values for logging:
            ``{"task": ..., "feature": ..., "response": ..., "total": ...}``.
        """
        # --- Task loss ---
        l_task = self.ce_loss(student_logits, labels)

        # --- Feature distillation loss ---
        l_feature = self.mse_loss(student_features, teacher_features.detach())

        # --- Response distillation (Hinton KD) ---
        t_soft = F.log_softmax(student_logits / self.T, dim=1)
        s_soft = F.softmax(teacher_logits.detach() / self.T, dim=1)
        l_response = F.kl_div(t_soft, s_soft, reduction="batchmean") * (self.T ** 2)

        # --- Combine ---
        total = (
            (1.0 - self.alpha) * l_task
            + self.alpha * self.beta * l_feature
            + self.alpha * (1.0 - self.beta) * l_response
        )

        components = {
            "task": l_task.item(),
            "feature": l_feature.item(),
            "response": l_response.item(),
            "total": total.item(),
        }
        return total, components


# ---------------------------------------------------------------------------
# Model factory helpers
# ---------------------------------------------------------------------------

def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained_teacher: bool = True,
    device: str | torch.device = "cpu",
) -> SpecCompressNet:
    """Instantiate and return a ``SpecCompressNet`` on the given device.

    Parameters
    ----------
    num_classes:
        Number of output classes.
    pretrained_teacher:
        Load ImageNet weights for the Teacher backbone.
    device:
        Target device string (e.g. ``"cuda"`` or ``"cpu"``).

    Returns
    -------
    SpecCompressNet
    """
    model = SpecCompressNet(num_classes=num_classes,
                            pretrained_teacher=pretrained_teacher)
    return model.to(device)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return trainable and total parameter counts for a model.

    Parameters
    ----------
    model:
        Any ``nn.Module``.

    Returns
    -------
    dict with keys ``"trainable"`` and ``"total"``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total}
