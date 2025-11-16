# Deformable Reference Detection

**Swin-Tiny + Deformable Attention for Reference-Based Detection**

Target: **IoU > 0.90** | Params: **~28M** | Speed: **~3x faster**

---

## 🎯 Architecture

```
Input: 3 Templates (3×640×640) + Search (3×640×640)
    ↓
Swin-Tiny Backbone (shared)
    → S2(256×80×80), S3(256×40×40), S4(256×20×20)
    ↓
Template Encoder → 9 tokens (3 templates × 3 scales)
    ↓
Deformable Decoder (6 layers) ← Search features
    → Self-attn → Template cross-attn → Deformable attn → FFN
    ↓
5 Predictions (logits + bboxes)
    ↓
Hungarian Matching → Best prediction
```

**Key Features:**
- Multi-scale deformable attention (3 levels × 4 points = 12 samples)
- Template-conditioned query initialization
- Focal loss + L1 + GIoU losses

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
conda activate aivn
pip install -r requirements.txt
```

### 2. Prepare Dataset
```
data/
├── train/
│   ├── templates/          # Template images
│   └── search/
│       ├── images/         # Search images
│       └── labels/         # YOLO format: cls cx cy w h
└── val/                    # Same structure
```

### 3. Train
```bash
# Recommended command
conda activate aivn
python train.py \
  --data_dir data/ \
  --output_dir outputs/ \
  --pretrained_backbone \
  --batch_size 32 \
  --workers 12 \
  --augment_prob 0.1

# Full command with all parameters
python train.py \
  --data_dir data/ \
  --output_dir outputs/ \
  --img_size 640 \
  --num_queries 5 \
  --hidden_dim 256 \
  --num_decoder_layers 6 \
  --num_heads 8 \
  --dim_feedforward 1024 \
  --dropout 0.1 \
  --num_points 4 \
  --pretrained_backbone \
  --loss_ce_weight 1.0 \
  --loss_bbox_weight 5.0 \
  --loss_giou_weight 2.0 \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --workers 12 \
  --augment_prob 0.1 \
  --seed 42
```

### 4. Resume Training
```bash
python train.py \
  --data_dir data/ \
  --output_dir outputs/ \
  --checkpoint_path outputs/checkpoint_epoch_50.pth \
  --pretrained_backbone
```

### 5. Inference
```bash
python inference.py \
  --checkpoint outputs/best.pth \
  --data_dir data/ \
  --split test \
  --output predictions.json
```

---

## ⚙️ Data Augmentation (Optimized)

**Template: KHÔNG augment** (resize + normalize only)

**Search: Augment tối ưu** (prob=0.1)

| Type | Parameter | Range | Note |
|------|-----------|-------|------|
| Geometric | Rotation | ±3° | Giảm từ ±5° |
| | Flip H | 50% | - |
| | Scale | 0.93-1.07 | Tiny objects |
| | Translate | ±3% | X, Y độc lập |
| Color | Brightness | 0.75-1.25 | Từ 0.7-1.3 |
| | Contrast | 0.75-1.25 | Từ 0.7-1.3 |
| | Saturation | 0.8-1.2 | Từ 0.7-1.3 |
| | Hue | ±0.03 | ⚠️ Từ ±0.05 |
| Other | Blur | 30% | σ=0.5-2.0 |
| | Noise | 15% | σ=0.05 |
| | Cutout | 20% | 2-5% |

**Lý do giảm color aug:** Tránh template-search mismatch → IoU tốt hơn

