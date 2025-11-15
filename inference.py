"""Inference script for Deformable RefDet."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model
from utils import build_dataset, collate_fn


def inference(model, loader, device):
    """Run inference on dataset."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for templates, searches, targets in tqdm(loader, desc="Inference"):
            templates = templates.to(device)
            searches = searches.to(device)
            
            # Forward pass
            pred_logits, pred_boxes = model(templates, searches)
            
            # Get best prediction per image
            batch_size = pred_boxes.shape[0]
            for i in range(batch_size):
                # Best prediction (highest confidence)
                scores = pred_logits[i].sigmoid().squeeze(-1)  # (num_queries,)
                best_idx = scores.argmax()
                
                best_score = scores[best_idx].item()
                best_box = pred_boxes[i, best_idx].cpu().numpy().tolist()
                
                predictions.append({
                    'bbox': best_box,  # [cx, cy, w, h] normalized
                    'confidence': best_score
                })
    
    return predictions


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = build_model(
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        num_decoder_layers=args.num_decoder_layers,
        pretrained_backbone=False
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    dataset = build_dataset(
        root=args.data_dir,
        split=args.split,
        augment=False,
        img_size=args.img_size
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn
    )
    
    # Run inference
    print(f"Running inference on {len(dataset)} samples...")
    predictions = inference(model, loader, device)
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Deformable RefDet")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    
    parser.add_argument("--num_queries", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    
    args = parser.parse_args()
    main(args)
