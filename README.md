# SpecCompress-Net 🔬

**Deepfake Detection Under Heavy Video Compression via Frequency-Domain Knowledge Distillation**

> IRC 2026 Research Project

---

## Overview

SpecCompress-Net addresses a critical blind spot in deepfake detection: **most detectors fail when video is heavily compressed** (high CRF values in H.264/H.265). Our approach makes two key innovations:

1. **Frequency Domain (FFT)** — Instead of operating on pixels, we analyse the 2-D Fourier magnitude spectrum of each frame. Compression artefacts leave distinct signatures in the frequency domain that survive even at very high CRF.

2. **Teacher-Student Knowledge Distillation** — A powerful ResNet-50 Teacher trained on uncompressed data transfers its representational knowledge to a lightweight Student CNN that operates on *compressed* FFT spectra. At inference only the Student runs, making deployment fast and practical.

```
┌─────────────────────────────────────────────────────────────┐
│                   Training Phase                            │
│                                                             │
│  Uncompressed Frame ──► Teacher (ResNet-50) ──► Soft Labels │
│                                    │                        │
│                              Feature KD Loss                │
│                                    │                        │
│  Compressed FFT  ──►  Student (LightCNN)  ──► Hard Labels  │
└─────────────────────────────────────────────────────────────┘
          ▼ Inference Phase ▼
  Compressed FFT  ──►  Student only  ──►  Real / Fake
```

---

## Architecture

| Component | Details |
|-----------|---------|
| **Teacher** | ResNet-50 (ImageNet), fine-tune `layer4` only |
| **Student** | Custom depthwise-separable CNN, ~1.2 M params |
| **KD Loss** | Task CE + Feature MSE + Response KL-div |
| **Input** | 224 × 224 log-magnitude FFT spectrum |

---

## Project Structure

```
SpecCompress-Net/
├── preprocess.py      # FFT pipeline, compression simulation, transforms
├── model.py           # Teacher, Student, SpecCompressNet, KD loss
├── train.py           # Training loop, checkpointing, CLI
├── visualize.py       # Comparison & spectrum grid figures
├── utils.py           # Frame extraction, evaluation metrics, helpers
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```python
from utils import extract_frames

# Extract 2 fps from each video
extract_frames("videos/real/vid001.mp4",  "data/real/",  fps=2, max_frames=1000)
extract_frames("videos/fake/vid001.mp4",  "data/fake/",  fps=2, max_frames=1000)
```

### 3. Train

```bash
python train.py \
    --real_dir   data/real   \
    --fake_dir   data/fake   \
    --output_dir checkpoints \
    --epochs     30          \
    --batch_size 32          \
    --lr         1e-4
```

### 4. Visualise

```bash
python visualize.py \
    --real  data/real/frame_0000001.jpg \
    --fake  data/fake/frame_0000001.jpg \
    --output figures/comparison.png
```

### 5. Evaluate

```python
from utils import evaluate_model, get_device
from model import SpecCompressNet
import torch

device = get_device()
model = SpecCompressNet()
ckpt = torch.load("checkpoints/speccompressnet_best_ep030.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.student.to(device)

results = evaluate_model(model.student, val_loader, device)
```

---

## Google Colab

```python
# Clone and install
!git clone https://github.com/<your-username>/SpecCompress-Net
%cd SpecCompress-Net
!pip install -r requirements.txt

# Quick sanity check
import torch
from model import build_model, count_parameters

model = build_model(device="cpu")
print(count_parameters(model.teacher))   # {'trainable': ..., 'total': ~25M}
print(count_parameters(model.student))  # {'trainable': ..., 'total': ~1.2M}

# Forward pass test
x_t = torch.randn(2, 3, 224, 224)
x_s = torch.randn(2, 3, 224, 224)
t_logits, t_feat, s_logits, s_feat = model(x_t, x_s)
print(t_logits.shape, s_logits.shape)   # (2, 2) (2, 2)
```

---

## Knowledge Distillation Loss

$$
\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\text{task}}
+ \alpha\beta\,\mathcal{L}_{\text{feature}}
+ \alpha(1-\beta)\,T^{2}\,\text{KL}\!\left(\sigma\!\left(\frac{z_s}{T}\right)\Big\|\sigma\!\left(\frac{z_t}{T}\right)\right)
$$

| Symbol | Default | Meaning |
|--------|---------|---------|
| α | 0.5 | Distillation weight |
| β | 0.3 | Feature vs response split |
| T | 4.0 | Softening temperature |

---

## Citation

```bibtex
@misc{speccompressnet2026,
  title   = {SpecCompress-Net: Deepfake Detection Under Heavy Video Compression
             via Frequency-Domain Knowledge Distillation},
  year    = {2026},
  note    = {IRC 2026}
}
```

---

## License

MIT
