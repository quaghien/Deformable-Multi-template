"""Dataset for Deformable RefDet (adapted from CA-DETR)."""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
import cv2

from .transforms import build_transforms, transform_bbox


class ReferenceDetectionDataset(Dataset):
    """
    Dataset for reference-based object detection with 3 templates.
    
    Each sample contains:
    - 3 templates: Reference images of the object
    - search: Search image where object should be detected
    - bbox: Ground truth bounding box in [cx, cy, w, h] format (normalized 0-1)
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        augment: bool = True,
        augment_prob: float = 0.5,
        img_size: int = 640
    ):
        """
        Args:
            root: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            augment: Enable augmentation
            augment_prob: Probability of applying augmentation
            img_size: Image size (square)
        """
        self.root = Path(root)
        self.split = split
        self.augment = augment and split == "train"
        self.augment_prob = augment_prob
        self.img_size = img_size
        
        # Paths
        self.template_dir = self.root / split / "templates"
        self.search_images_dir = self.root / split / "search" / "images"
        self.search_labels_dir = self.root / split / "search" / "labels"
        
        # Collect data
        self.template_paths = self._collect_templates()
        self.samples = self._collect_samples()
        
        # Build transforms - Tách riêng template vs search
        self.template_transform = build_transforms(img_size=img_size, augment=False)  # Template: không augment
        self.search_transform = build_transforms(img_size=img_size, augment=self.augment)  # Search: có augment
    
    @staticmethod
    def _extract_video_id(filename: str) -> str:
        """Extract video_id from filename."""
        name = Path(filename).stem
        parts = name.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return parts[0]
    
    def _collect_templates(self) -> Dict[str, List[Path]]:
        """Collect templates grouped by video_id."""
        templates = {}
        template_files = list(self.template_dir.glob("*.jpg")) + list(self.template_dir.glob("*.png"))
        
        for template_path in template_files:
            video_id = self._extract_video_id(template_path.name)
            if video_id not in templates:
                templates[video_id] = []
            templates[video_id].append(template_path)
        
        if not templates:
            raise RuntimeError(f"No templates found in {self.template_dir}")
        return templates
    
    def _collect_samples(self) -> List[Tuple[str, Path, Path]]:
        """Collect samples (video_id, image_path, label_path)."""
        samples = []
        image_files = sorted(self.search_images_dir.glob("*.jpg")) + sorted(self.search_images_dir.glob("*.png"))
        
        for img_path in image_files:
            label_path = self.search_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                video_id = self._extract_video_id(img_path.name)
                if video_id in self.template_paths:
                    samples.append((video_id, img_path, label_path))
        
        if not samples:
            raise RuntimeError(f"No samples found in {self.search_images_dir}")
        
        return samples
    
    def _parse_label(self, label_path: Path) -> Tuple[float, float, float, float]:
        """
        Parse label file to get bbox coordinates.
        
        Format: class_id x_center y_center width height (normalized 0-1)
        
        Returns:
            (x_c, y_c, w, h): Normalized bbox coordinates
        """
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                raise ValueError(f"Empty label file: {label_path}")
            
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid label format in {label_path}: {line}")
            
            # Parse: class_id x_c y_c w h
            x_c, y_c, w, h = map(float, parts[1:5])
            
            # Clamp to valid range
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w = max(1e-6, min(1.0, w))
            h = max(1e-6, min(1.0, h))
            
            return x_c, y_c, w, h
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dict with:
            - templates: (3, 3, H, W) - 3 template image tensors
            - search: (3, H, W) - Search image tensor
            - bbox: (4,) - Ground truth bbox [cx, cy, w, h] (normalized)
            - label: (1,) - Class label (always 1 for single-object detection)
        """
        video_id, img_path, label_path = self.samples[idx]
        
        # Get all templates for this video
        template_paths = self.template_paths[video_id]
        
        # Always select exactly 3 templates (repeat if necessary)
        if len(template_paths) >= 3:
            selected_templates = random.sample(template_paths, 3)
        else:
            # Repeat templates to get 3
            selected_templates = template_paths * (3 // len(template_paths) + 1)
            selected_templates = selected_templates[:3]
        
        # Parse label
        x_c, y_c, w, h = self._parse_label(label_path)
        
        # Decide whether to augment SEARCH image (template luôn không augment)
        should_augment = self.augment and random.random() < self.augment_prob
        
        # Generate augmentation parameters cho SEARCH IMAGE
        # Version tối ưu: cân bằng giữa diversity và template-search consistency
        if should_augment:
            aug_params = {
                # Geometric augmentations
                'angle': random.uniform(-3, 3),          # ±3° (giảm từ ±5°)
                'flip_h': random.random() < 0.5,         # 50% horizontal flip
                'scale': random.uniform(0.93, 1.07),     # Scale jitter cho tiny objects
                'translate_x': random.uniform(-0.03, 0.03),  # ±3% translation X
                'translate_y': random.uniform(-0.03, 0.03),  # ±3% translation Y
                # Color augmentations (giảm cường độ để tránh mismatch)
                'brightness': random.uniform(0.75, 1.25),  # 0.75-1.25 (từ 0.7-1.3)
                'contrast': random.uniform(0.75, 1.25),    # 0.75-1.25 (từ 0.7-1.3)
                'saturation': random.uniform(0.8, 1.2),    # 0.8-1.2 (từ 0.7-1.3)
                'hue': random.uniform(-0.03, 0.03),        # ±0.03 (từ ±0.05) - QUAN TRỌNG
                # Other augmentations
                'blur': random.random() < 0.3,
                'noise': random.random() < 0.15,     # Giảm từ 0.2 → 0.15
                'cutout': random.random() < 0.2,
            }
        else:
            aug_params = None  # Không augment → None (quan trọng!)
        
        # Transform 3 templates - KHÔNG BAO GIỜ augment
        template_tensors = []
        for template_path in selected_templates:
            template_tensor = self.template_transform(template_path, aug_params=None)
            template_tensors.append(template_tensor)
        
        # Stack templates: (3, 3, H, W)
        templates = torch.stack(template_tensors)
        
        # Transform search image - Augment theo aug_params
        search_tensor = self.search_transform(img_path, aug_params=aug_params)
        
        # Transform bbox to match search image augmentation (scale, translate, rotate, flip)
        if aug_params is not None:
            x_c_aug, y_c_aug, w_aug, h_aug = transform_bbox(
                x_c, y_c, w, h,
                img_size=self.img_size,
                angle=aug_params['angle'],
                flip_h=aug_params['flip_h'],
                flip_v=False,  # Không dùng vertical flip
                scale=aug_params['scale'],
                translate_x=aug_params['translate_x'],
                translate_y=aug_params['translate_y']
            )
        else:
            # Không augment → bbox giữ nguyên
            x_c_aug, y_c_aug, w_aug, h_aug = x_c, y_c, w, h
        
        return {
            'templates': templates,  # (3, 3, H, W)
            'search': search_tensor,  # (3, H, W)
            'bbox': torch.tensor([x_c_aug, y_c_aug, w_aug, h_aug], dtype=torch.float32),
            'label': torch.tensor([1], dtype=torch.int64)
        }


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        templates: (B, 3, 3, H, W) - Batched template images
        searches: (B, 3, H, W) - Batched search images
        targets: List of dicts with 'boxes' (1, 4) and 'labels' (1,)
    """
    templates = torch.stack([item['templates'] for item in batch])  # (B, 3, 3, H, W)
    searches = torch.stack([item['search'] for item in batch])  # (B, 3, H, W)
    
    targets = []
    for item in batch:
        targets.append({
            'boxes': item['bbox'].unsqueeze(0),  # (1, 4)
            'labels': item['label']  # (1,)
        })
    
    return templates, searches, targets


def build_dataset(
    root: str,
    split: str = "train",
    augment: bool = True,
    augment_prob: float = 0.5,
    img_size: int = 640
) -> ReferenceDetectionDataset:
    """
    Build dataset.
    
    Args:
        root: Root directory of dataset
        split: Dataset split
        augment: Enable augmentation
        augment_prob: Augmentation probability
        img_size: Image size
        
    Returns:
        dataset: ReferenceDetectionDataset instance
    """
    return ReferenceDetectionDataset(
        root=root,
        split=split,
        augment=augment,
        augment_prob=augment_prob,
        img_size=img_size
    )
