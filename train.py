"""Training script for Deformable Reference Detection."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model import build_model, build_criterion
from utils import build_dataset, collate_fn


def train_one_epoch(model, criterion, loader, optimizer, scheduler, device, epoch, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    use_amp = scaler is not None
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]{'[FP16]' if use_amp else '[FP32]'}", ncols=120)
    for templates, searches, targets in pbar:
        templates = templates.to(device)
        searches = searches.to(device)
        
        # Move targets to device
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with optional autocast
        if use_amp:
            with autocast():
                pred_logits, pred_boxes = model(templates, searches)
                
                # Prepare outputs dict
                outputs = {
                    'pred_logits': pred_logits,
                    'pred_boxes': pred_boxes
                }
                
                # Compute losses
                losses = criterion(outputs, targets)
                loss = sum(losses.values())
        else:
            pred_logits, pred_boxes = model(templates, searches)
            
            # Prepare outputs dict
            outputs = {
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes
            }
            
            # Compute losses
            losses = criterion(outputs, targets)
            loss = sum(losses.values())
        
        # Backward pass with optional GradScaler
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_loss_ce += losses.get('loss_ce', 0).item()
        total_loss_bbox += losses.get('loss_bbox', 0).item()
        total_loss_giou += losses.get('loss_giou', 0).item()
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ce": f"{losses.get('loss_ce', 0).item():.4f}",
            "bbox": f"{losses.get('loss_bbox', 0).item():.4f}",
            "giou": f"{losses.get('loss_giou', 0).item():.4f}",
            "lr": f"{current_lr:.2e}"
        })
    
    steps = len(loader)
    return {
        "loss": total_loss / steps,
        "loss_ce": total_loss_ce / steps,
        "loss_bbox": total_loss_bbox / steps,
        "loss_giou": total_loss_giou / steps,
    }


def compute_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes in [cx, cy, w, h] format."""
    # Convert to [x1, y1, x2, y2]
    b1_x1, b1_y1 = boxes1[:, 0] - boxes1[:, 2] / 2, boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2, b1_y2 = boxes1[:, 0] + boxes1[:, 2] / 2, boxes1[:, 1] + boxes1[:, 3] / 2
    b2_x1, b2_y1 = boxes2[:, 0] - boxes2[:, 2] / 2, boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2, b2_y2 = boxes2[:, 0] + boxes2[:, 2] / 2, boxes2[:, 1] + boxes2[:, 3] / 2
    
    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    
    return inter_area / (union_area + 1e-6)


def evaluate(model, criterion, loader, device, epoch):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    total_iou = 0.0
    total_iou_best_of_all = 0.0  # Best IoU among all queries (model potential)
    num_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", ncols=120)
        for templates, searches, targets in pbar:
            templates = templates.to(device)
            searches = searches.to(device)
            
            # Move targets to device
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            
            # Forward pass
            pred_logits, pred_boxes = model(templates, searches)
            
            # Prepare outputs dict
            outputs = {
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes
            }
            
            # Compute losses
            losses = criterion(outputs, targets)
            
            # Total loss
            loss = sum(losses.values())
            
            # Accumulate losses
            total_loss += loss.item()
            total_loss_ce += losses.get('loss_ce', 0).item()
            total_loss_bbox += losses.get('loss_bbox', 0).item()
            total_loss_giou += losses.get('loss_giou', 0).item()
            
            # Compute IoU for best prediction
            batch_size = pred_boxes.shape[0]
            for i in range(batch_size):
                gt_box = targets[i]['boxes'][0].unsqueeze(0)  # (1, 4)
                
                # Method 1: Chọn query theo HIGHEST CONFIDENCE (giống inference)
                best_idx = pred_logits[i].sigmoid().argmax()
                pred_box = pred_boxes[i, best_idx].unsqueeze(0)
                iou_by_conf = compute_iou(pred_box, gt_box).item()
                
                # Method 2: Best IoU among all queries (upper bound performance)
                ious = []
                for j in range(pred_boxes.shape[1]):
                    pred_box = pred_boxes[i, j].unsqueeze(0)
                    iou = compute_iou(pred_box, gt_box).item()
                    ious.append(iou)
                iou_best = max(ious)
                
                total_iou += iou_by_conf  # IoU của query được chọn (realistic)
                total_iou_best_of_all += iou_best  # IoU tốt nhất có thể (potential)
                num_samples += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "iou": f"{total_iou/num_samples:.4f}",
                "iou_max": f"{total_iou_best_of_all/num_samples:.4f}"
            })
    
    steps = len(loader)
    return {
        "loss": total_loss / steps,
        "loss_ce": total_loss_ce / steps,
        "loss_bbox": total_loss_bbox / steps,
        "loss_giou": total_loss_giou / steps,
        "mean_iou": total_iou / num_samples if num_samples > 0 else 0.0,
        "mean_iou_best": total_iou_best_of_all / num_samples if num_samples > 0 else 0.0,
    }


def main(args):
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        import numpy as np
        import random
        np.random.seed(args.seed)
        random.seed(args.seed)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {args.seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build model
    print(f"\nBuilding Deformable RefDet model...")
    model = build_model(
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_levels=3,  # S2, S3, S4
        num_points=args.num_points,
        pretrained_backbone=args.pretrained_backbone
    )
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.2f}M")
    
    # Build criterion
    weight_dict = {
        'loss_ce': args.loss_ce_weight,
        'loss_bbox': args.loss_bbox_weight,
        'loss_giou': args.loss_giou_weight
    }
    criterion = build_criterion(
        num_classes=1,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    criterion.to(device)
    
    # Build datasets
    print(f"\nLoading datasets from: {args.data_dir}")
    train_dataset = build_dataset(
        root=args.data_dir,
        split="train",
        augment=True,
        augment_prob=args.augment_prob,
        img_size=args.img_size
    )
    val_dataset = build_dataset(
        root=args.data_dir,
        split="val",
        augment=False,
        img_size=args.img_size
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"\n🎨 Data Augmentation:")
    print(f"  Train: ENABLED (prob={args.augment_prob})")
    print(f"    - Rotation: ±5°")
    print(f"    - Horizontal flip: 50%")
    print(f"    - Brightness: 0.8-1.2")
    print(f"    - Contrast: 0.8-1.2")
    print(f"    - Saturation: 0.8-1.2")
    print(f"    - BBox transform: ENABLED (synced with image)")
    print(f"  Val: DISABLED")
    
    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Setup mixed precision training
    scaler = None
    if args.mixed_precision and torch.cuda.is_available():
        print(f"\n🚀 Mixed Precision: ENABLED (FP16)")
        scaler = GradScaler()
        # Convert model to FP16 for training and saving
        model = model.half()
        print(f"   ✓ Model converted to FP16")
    else:
        print(f"\n🚀 Mixed Precision: DISABLED (FP32)")
    
    # Build scheduler
    if args.lr_schedule == 'cosine':
        total_steps = args.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.min_lr
        )
    else:
        scheduler = None
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_iou = 0.0
    
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        print(f"\nLoading checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # Load model weights and convert FP32→FP16 if needed
        model_state = checkpoint['model']
        
        # Check if we need to convert FP32 checkpoint to FP16
        if args.mixed_precision:
            checkpoint_dtype = next(iter(model_state.values())).dtype
            if checkpoint_dtype == torch.float32:
                print(f"   📋 Converting FP32 checkpoint to FP16 for training")
                # Convert FP32 checkpoint to FP16
                model_state = {k: v.half() if v.dtype == torch.float32 else v for k, v in model_state.items()}
                print(f"✓ Loaded model weights from checkpoint (FP32 → FP16)")
            else:
                print(f"✓ Loaded model weights from checkpoint (FP16 → FP16)")
        else:
            print(f"✓ Loaded model weights from checkpoint (FP32 mode)")
        
        model.load_state_dict(model_state)
        
        # Optionally load optimizer/scheduler state (only if reset_optimizer is False)
        if not args.reset_optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"✓ Loaded optimizer state")
            
            # Load GradScaler state if available and using mixed precision
            if args.mixed_precision and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
                print(f"✓ Loaded GradScaler state")
            
            if scheduler is not None and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print(f"✓ Loaded scheduler state")
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_iou = checkpoint.get('best_iou', 0.0)
            print(f"✓ Resuming from epoch {start_epoch}, best IoU: {best_iou:.4f}")
        else:
            # Train from scratch with loaded weights
            print(f"✓ Training from epoch 0 with loaded model weights (optimizer reset)")
            start_epoch = 0
            best_iou = 0.0
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    history = []
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, device, epoch, scaler
        )
        
        # Validate
        val_metrics = evaluate(model, criterion, val_loader, device, epoch)
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics
        }
        history.append(metrics)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"CE: {train_metrics['loss_ce']:.4f}, "
              f"BBox: {train_metrics['loss_bbox']:.4f}, "
              f"GIoU: {train_metrics['loss_giou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"IoU: {val_metrics['mean_iou']:.4f} "
              f"(best_possible: {val_metrics['mean_iou_best']:.4f})")
        
        # Debug: Show gap between actual vs potential
        iou_gap = val_metrics['mean_iou_best'] - val_metrics['mean_iou']
        if iou_gap > 0.1 and epoch > 20:
            print(f"  ⚠️  IoU gap: {iou_gap:.4f} - Model not selecting best queries! "
                  f"Classification head needs more training.")
        
        # Debug: Show if IoU is too low
        if val_metrics['mean_iou'] < 0.1 and epoch < 10:
            print(f"  ⚠️  IoU very low! Check: 1) --pretrained_backbone flag, 2) data labels, 3) bbox normalization")
        
        # Save best checkpoint
        if val_metrics['mean_iou'] > best_iou:
            best_iou = val_metrics['mean_iou']
            
            # Prepare model state - ensure FP16 if using mixed precision
            model_state = model.state_dict()
            if args.mixed_precision:
                # Ensure all weights are FP16
                model_state = {k: v.half() if v.dtype == torch.float32 else v for k, v in model_state.items()}
            
            checkpoint = {
                'model': model_state,
                'epoch': epoch,
                'best_iou': best_iou,
                'mixed_precision': args.mixed_precision
            }
            
            # Add GradScaler state if using mixed precision
            if args.mixed_precision and scaler is not None:
                checkpoint['scaler'] = scaler.state_dict()
            
            torch.save(checkpoint, output_dir / "best.pth")
            precision_str = "FP16" if args.mixed_precision else "FP32"
            print(f"  → Saved best checkpoint ({precision_str}, IoU: {best_iou:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            # Prepare model state - ensure FP16 if using mixed precision
            model_state = model.state_dict()
            if args.mixed_precision:
                model_state = {k: v.half() if v.dtype == torch.float32 else v for k, v in model_state.items()}
            
            checkpoint = {
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_iou': best_iou,
                'mixed_precision': args.mixed_precision
            }
            
            # Add scheduler and scaler states
            if scheduler is not None:
                checkpoint['scheduler'] = scheduler.state_dict()
            if args.mixed_precision and scaler is not None:
                checkpoint['scaler'] = scaler.state_dict()
            
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
        
        # Save last epoch checkpoint (always overwrite)
        model_state = model.state_dict()
        if args.mixed_precision:
            model_state = {k: v.half() if v.dtype == torch.float32 else v for k, v in model_state.items()}
        
        last_checkpoint = {
            'model': model_state,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_iou': best_iou,
            'mixed_precision': args.mixed_precision
        }
        
        if scheduler is not None:
            last_checkpoint['scheduler'] = scheduler.state_dict()
        if args.mixed_precision and scaler is not None:
            last_checkpoint['scaler'] = scaler.state_dict()
            
        torch.save(last_checkpoint, output_dir / f"last_epoch_{epoch+1}.pth")
        
        # Save history
        with open(output_dir / "history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    precision_str = "FP16" if args.mixed_precision else "FP32"
    print(f"\nTraining complete! Best IoU: {best_iou:.4f} ({precision_str})")
    print(f"Last epoch: {args.epochs}")
    print(f"Checkpoints saved to: {output_dir}")
    if args.mixed_precision:
        print(f"📁 All checkpoints saved in FP16 format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deformable RefDet")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=640)
    
    # Model
    parser.add_argument("--num_queries", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_points", type=int, default=4)
    parser.add_argument("--pretrained_backbone", action="store_true")
    
    # Loss
    parser.add_argument("--loss_ce_weight", type=float, default=1.0)
    parser.add_argument("--loss_bbox_weight", type=float, default=5.0)
    parser.add_argument("--loss_giou_weight", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--augment_prob", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true", 
                        help="Enable FP16 mixed precision training")
    
    # Checkpointing
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--save_every", type=int, default=10)
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    
    args = parser.parse_args()
    main(args)
