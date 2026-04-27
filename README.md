---
title: PCB Defect Detection (RT-DETRv4)
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# PCB Defect Detection — RT-DETRv4 X on DsPCBSD+

RT-DETRv4 X fine-tuned on the DsPCBSD+ dataset for 9-class copper-layer defect detection.

## Live Demo

Try it in your browser without any setup:

**[https://huggingface.co/spaces/mcthebest/PCB_RTDETR](https://huggingface.co/spaces/mcthebest/PCB_RTDETR)**

Upload your own PCB image or pick from 29 built-in test images. The confidence threshold is adjustable in the sidebar.

## Model

### Architecture: RT-DETRv4 X

RT-DETRv4 builds on the RT-DETR lineage (v1, v2, v3) with ideas from D-FINE and DEIM. The main addition is a semantic distillation framework that uses a frozen DINOv3 teacher during training only, adding no cost at inference.

| Component | Detail |
|---|---|
| Backbone | HGNetv2-X, multi-scale features at stride 8/16/32 |
| Encoder | Efficient Hybrid Encoder: AIFI (global self-attention on S5) + CCFF (CNN cross-scale fusion) |
| Decoder | DFINETransformer, 6 layers, 300 queries, 32-bin probabilistic box regression per edge |
| Teacher (train only) | DINOv3 ViT-B/16 (frozen), trained on LVD-1689M (~1.7B images) |
| DSI module | Aligns F5 features with DINOv3 semantics via cosine similarity loss |
| GAM module | Adjusts DSI loss weight per epoch based on gradient norms |

Training loss:

```
L_total = L_det + λ · L_DSI
```

`L_det` includes VFL, L1, GIoU, FGL, DDF, and MAL losses from D-FINE and DEIM. DINOv3 is not loaded at inference.

### Training Configuration

| Parameter | Value |
|---|---|
| Dataset | DsPCBSD+ (8,208 train / 2,051 val, 80-20 split) |
| Pretrained weights | RT-DETRv4-X COCO + DINOv3 ViT-B/16 (LVD-142M) |
| Input resolution | 640 x 640 px |
| Epochs | 10 (72 recommended; limited by Colab T4) |
| Batch size | 8 |
| Optimizer | AdamW (lr encoder/decoder = 2e-4, lr backbone = 1e-5, wd = 1e-4) |
| LR Scheduler | FlatCosine (warmup 500 iter, flat 5 epoch, no-aug 2 epoch) |
| Mixed precision | AMP FP16/FP32 |
| Hardware | NVIDIA Tesla T4 (16 GB VRAM), Google Colab |

Mosaic and MixUp augmentation were disabled due to memory constraints. Multi-scale training (480-800 px, 11 levels) was kept.

## Results on DsPCBSD+ Validation Set

### Overall COCO Metrics

| Metric | Value |
|---|---|
| mAP @ IoU=0.50 | 0.863 |
| mAP @ IoU=0.50:0.95 | 0.522 |
| mAP @ IoU=0.75 | 0.551 |
| AP small (<32^2 px) | 0.439 |
| AP medium | 0.602 |
| AP large (>96^2 px) | 0.754 |
| AR @ maxDets=100 | 0.686 |

### Per-Class AP @ IoU=0.50:0.95

| Class | Full Name | AP |
|---|---|---|
| SH | Short Circuit | 0.597 |
| SP | Spur (Copper Spike) | 0.381 |
| SC | Spurious Copper | 0.511 |
| OP | Open Circuit | 0.555 |
| MB | Mouse Bite | 0.408 |
| HB | Hole Breakout | 0.807 |
| CS | Conductor Scratch | 0.504 |
| CFO | Conductor Foreign Object | 0.462 |
| BMFO | Base Material Foreign Object | 0.478 |
| Mean | | 0.522 |

### Per-Class F1 @ Conf>=0.5, IoU>=0.5

| Class | Precision | Recall | F1 |
|---|---|---|---|
| SH | 0.91 | 0.86 | 0.88 |
| SP | 0.77 | 0.69 | 0.73 |
| SC | 0.89 | 0.77 | 0.83 |
| OP | 0.85 | 0.84 | 0.84 |
| MB | 0.87 | 0.69 | 0.77 |
| HB | 0.91 | 0.93 | 0.92 |
| CS | 0.82 | 0.65 | 0.73 |
| CFO | 0.86 | 0.56 | 0.68 |
| BMFO | 0.87 | 0.83 | 0.85 |
| Mean | | | 0.802 |

### Comparison with SOTA on DsPCBSD+

| Model | Backbone | mAP@50 | mAP@50:95 | Notes |
|---|---|---|---|---|
| YOLOv11-CGL | YOLOv11n | 84.5% | 51.6% | 300 epochs, lightweight |
| PCB-AM | YOLOv8s | 85.7% | N/A | Attention-guided modules |
| PCB-FS | YOLOv8 | 86.2% | 52.4% | Frequency-spatial features |
| RT-DETRv4 X (ours) | HGNetv2-X + DINOv3 | 86.3% | 52.25% | 10 epochs, T4 only |

RT-DETRv4 X matches or exceeds all compared methods at only 10 epochs on a single T4. PCB-FS, for comparison, used 100+ epochs with more resources.

## Dataset: DsPCBSD+

DsPCBSD+ is a 2024 open dataset for PCB copper-layer defect detection, captured by a professional AOI system (AGLE'OL AOI-100 V8, 16K camera, controlled LED lighting). Nine defect categories, annotated at instance level:

| Code | Defect | Description |
|---|---|---|
| SH | Short | Conductive bridge between two traces |
| SP | Spur | Anomalous copper spike from a trace |
| SC | Spurious Copper | Unwanted copper on board surface |
| OP | Open Circuit | Break in a conductive trace |
| MB | Mouse Bite | Edge notch or chip in the trace |
| HB | Hole Breakout | Damage to material around a drill hole |
| CS | Conductor Scratch | Scratch mark on a conductive trace |
| CFO | Conductor Foreign Object | Foreign particle on a trace |
| BMFO | Base Material Foreign Object | Foreign particle in substrate material |

> S. Lv et al., "A dataset for deep learning based detection of printed circuit board surface defect," *Scientific Data*, vol. 11, no. 1, p. 811, 2024. https://doi.org/10.1038/s41597-024-03656-8

## Files

| File | Description |
|---|---|
| `last.pth` | Fine-tuned checkpoint (EMA weights) |
| `test_image/` | 29 example PCB images for inference testing |

The training code, config, and Streamlit app are in the companion Space:
[https://huggingface.co/spaces/mcthebest/PCB_RTDETR](https://huggingface.co/spaces/mcthebest/PCB_RTDETR)

## Running Locally

Clone the Space and install dependencies:

```bash
git clone https://huggingface.co/spaces/mcthebest/PCB_RTDETR
cd PCB_RTDETR
pip install -r requirements.txt
```

Download the checkpoint:

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="mcthebest/PCB_RTDETR",
    repo_type="model",
    filename="last.pth",
)
```

Or place it manually at `outputs/rtv4_hgnetv2_x_pcb/last.pth`, then run:

```bash
streamlit run app.py
```

## How Inference Works

The image is resized to 640x640 and passed through RT-DETRv4 X. The decoder outputs 300 candidate queries which the postprocessor filters by confidence threshold (default 0.30). Detections are returned as `(labels, boxes, scores)` and drawn with labeled bounding boxes. DINOv3 is not loaded at inference.

## Known Limitations

- Small defects (AP=0.439): sub-32px defects like SP (Spur) are the hardest class.
- Only 10 epochs were run vs. the recommended 72. The learning curve had not plateaued, so more training should meaningfully improve mAP@50:95.
- The dataset was collected under controlled AOI conditions. Images from different lighting or camera setups may need domain adaptation.
- AR@100=0.686, meaning roughly 31% of defects are missed. Not production-ready for zero-miss QC pipelines.

## References

- [RT-DETRv4](https://arxiv.org/abs/2510.25257) — Liao et al., 2025
- [D-FINE](https://arxiv.org/abs/2410.13842) — Peng et al., ICLR 2025
- [DEIM](https://arxiv.org/abs/2412.04234) — Huang et al., CVPR 2025
- [DINOv3](https://arxiv.org/abs/2508.10104) — Siméoni et al., 2025
- [RT-DETRv4 GitHub](https://github.com/RT-DETRs/RT-DETRv4)
- [DsPCBSD+ dataset](https://doi.org/10.6084/m9.figshare.24970329)