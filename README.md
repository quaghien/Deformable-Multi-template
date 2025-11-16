# Deformable Reference Detection

**Swin-Tiny + Deformable Attention for Multi-Template Single-Object Detection**

Target: **IoU > 0.90** | Params: **28M** | Memory: **FP16 ~5GB, FP32 ~10GB**

---

## 🏗️ Architecture (Chi tiết)

```
📸 INPUT STAGE
┌─────────────────────────────────────────────────────────────────────┐
│  Templates: 3×(3,640,640)                       Search: (3,640,640) │
│      ↓                                                    ↓         │
│  ┌─Template 1─┐  ┌─Template 2─┐  ┌─Template 3─┐    ┌─Search Image─┐ │
│  │ (3,640,640)│  │ (3,640,640)│  │ (3,640,640)│    │ (3,640,640)  │ │
│  └────────────┘  └────────────┘  └────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
🧠 BACKBONE (Swin-Tiny - SHARED across all images)
┌─────────────────────────────────────────────────────────────────────┐
│  Swin-Tiny Backbone (28M params, pretrained ImageNet)               │
│                                                                     │
│  Stage 1: (3,640,640) → Skip (lightweight)                          │
│  Stage 2: (96,160,160) → S2 Features (256,80,80)   [1/8 scale]      │
│  Stage 3: (192,80,80)  → S3 Features (256,40,40)   [1/16 scale]     │
│  Stage 4: (384,40,40)  → S4 Features (256,20,20)   [1/32 scale]     │
│                                                                     │
│  Output: Multi-scale features for Templates + Search                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
🎯 TEMPLATE ENCODING
┌─────────────────────────────────────────────────────────────────────┐
│  Template Features → Template Encoder                               │
│                                                                     │
│  Template 1: S2(256,80,80) + S3(256,40,40) + S4(256,20,20)          │
│              → 3 scale tokens → Global Average Pool → 3 tokens      │
│                                                                     │
│  Template 2: Same process → 3 tokens                                │
│  Template 3: Same process → 3 tokens                                │
│                                                                     │
│  Total: 3 templates × 3 scales = 9 Template Tokens (256 dim each)   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
🔄 DEFORMABLE DECODER (6 layers)
┌─────────────────────────────────────────────────────────────────────┐
│  Input: 5 Learnable Queries (256 dim each)                          │
│       + 9 Template Tokens (256 dim each)                            │
│       + Search Features: S2(256,80,80), S3(256,40,40), S4(256,20,20)│
│                                                                     │
│  Each Decoder Layer:                                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 1. Self-Attention (5 queries ↔ 5 queries)                   │    │
│  │ 2. Template Cross-Attention (5 queries ↔ 9 template tokens) │    │
│  │ 3. Deformable Search Attention:                             │    │
│  │    - Sample từ 3 levels (S2,S3,S4)                          │    │
│  │    - 4 points per level = 12 sampling points                │    │
│  │    - Learnable offsets + attention weights                  │    │
│  │ 4. FFN (256 → 2048 → 256)                                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ×6 layers                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
📊 PREDICTION HEADS
┌─────────────────────────────────────────────────────────────────────┐
│  5 Query Features (256 dim each)                                    │
│                    ↓                                                │
│  ┌─Classification Head─┐    ┌─Regression Head─┐                     │
│  │ Linear(256 → 1)     │    │ MLP(256→256→4)  │                     │
│  │ → 5 Logits          │    │ → 5 BBoxes      │                     │
│  └─────────────────────┘    └─────────────────┘                     │
│                                                                     │
│  Output: pred_logits(5,1) + pred_boxes(5,4)                         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
🎯 HUNGARIAN MATCHING
┌─────────────────────────────────────────────────────────────────────┐
│  5 Predictions vs 1 Ground Truth                                    │
│                                                                     │
│  Cost Matrix (5×1):                                                 │
│  - Classification cost: Focal Loss weight                           │
│  - L1 BBox cost: |pred_box - gt_box|                                │
│  - GIoU cost: 1 - GIoU(pred_box, gt_box)                            │
│                                                                     │
│  Hungarian Algorithm → Select Best Query                            │
│  Loss applied ONLY to selected query                                │
└─────────────────────────────────────────────────────────────────────┘

🏃‍♂️ INFERENCE: argmax(sigmoid(pred_logits)) → Select highest confidence query
```

---

## 🚀 Quick Start

### Setup
```bash
conda activate aivn
pip install -r requirements.txt
```

### Training (Production Config)
```bash
# Actual config đang dùng (FP16 mixed precision)
python train.py \
  --data_dir refdet/retrieval_dataset_flat_zoomed/ \
  --output_dir drive/MyDrive/ZALO2025 \
  --checkpoint_path drive/MyDrive/AIVN/ZALOAI2025/last_epoch_2.pth \
  --mixed_precision \
  --img_size 640 \
  --num_queries 5 \
  --hidden_dim 256 \
  --num_decoder_layers 6 \
  --num_heads 8 \
  --dim_feedforward 2048 \
  --dropout 0.1 \
  --num_points 4 \
  --pretrained_backbone \
  --batch_size 16 \
  --epochs 10 \
  --lr 2e-4 \
  --min_lr 7e-5 \
  --lr_schedule cosine \
  --weight_decay 1e-4 \
  --augment_prob 0.1 \
  --workers 12 \
  --save_every 5 \
  --seed 42
```

### Key Features
- ⚡ **FP16 Mixed Precision**: `--mixed_precision` (2x memory, 1.5x speed)
- 🔄 **Auto Checkpoint Conversion**: Load FP32 → Auto convert FP16 → Save FP16
- 🎨 **Smart Augmentation**: Template clean, Search augmented (prob=0.1)
- 🧠 **Pretrained Backbone**: Swin-Tiny ImageNet weights

---

## 📊 Performance

| Metric | Value | Note |
|--------|--------|------|
| Target IoU | >0.90 | Production ready |
| Parameters | 28M | Swin-Tiny backbone |
| Memory (FP16) | ~5GB | With batch_size=16 |
| Speed | ~15 min/epoch | A100, optimized |
| Convergence | ~50 epochs | With pretrained backbone |

---

## 🛠️ Critical Settings

```bash
--pretrained_backbone     # MUST use (IoU 0.9 vs 0.3 without)
--mixed_precision        # 50% memory reduction
--augment_prob 0.1       # Template stable, search augmented
--dim_feedforward 2048   # FFN expansion (vs default 1024)
--lr 2e-4               # Higher than default 1e-4
--min_lr 7e-5           # Cosine schedule floor
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
**Status:** ✅ Production Ready | **License:** MIT

