"""Image transformation utilities (adapted from CA-DETR)."""

import random
import math
from typing import Callable, Union, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load image from path or return if already numpy array."""
    if isinstance(image, np.ndarray):
        return image
    image = cv2.imread(str(image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return image


def transform_bbox(
    x_c: float, y_c: float, w: float, h: float,
    img_size: int,
    angle: float = 0.0,
    flip_h: bool = False,
    flip_v: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Transform bounding box coordinates according to geometric augmentations.
    
    Args:
        x_c, y_c, w, h: Normalized bbox coordinates (0-1) in format (center_x, center_y, width, height)
        img_size: Image size (assumed square)
        angle: Rotation angle in degrees (applied first)
        flip_h: Whether to flip horizontally (applied after rotation)
        flip_v: Whether to flip vertically (applied after flip_h)
        
    Returns:
        (new_x_c, new_y_c, new_w, new_h): Transformed normalized bbox coordinates
    """
    # Convert to pixel coordinates
    x_c_px = x_c * img_size
    y_c_px = y_c * img_size
    w_px = w * img_size
    h_px = h * img_size
    
    # Convert center-based to corner-based
    x1 = x_c_px - w_px / 2
    y1 = y_c_px - h_px / 2
    x2 = x_c_px + w_px / 2
    y2 = y_c_px + h_px / 2
    
    # Get 4 corners
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)
    
    center = np.array([img_size / 2, img_size / 2], dtype=np.float32)
    
    # 1. Rotation
    if abs(angle) > 1e-6:
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        corners = (corners - center) @ rot_matrix.T + center
    
    # 2. Horizontal flip
    if flip_h:
        corners[:, 0] = img_size - corners[:, 0]
    
    # 3. Vertical flip
    if flip_v:
        corners[:, 1] = img_size - corners[:, 1]
    
    # Convert back to center format
    x_min = corners[:, 0].min()
    y_min = corners[:, 1].min()
    x_max = corners[:, 0].max()
    y_max = corners[:, 1].max()
    
    # Clip to image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_size, x_max)
    y_max = min(img_size, y_max)
    
    # Convert to normalized center format
    new_x_c = (x_min + x_max) / 2 / img_size
    new_y_c = (y_min + y_max) / 2 / img_size
    new_w = (x_max - x_min) / img_size
    new_h = (y_max - y_min) / img_size
    
    # Ensure valid bbox
    new_w = max(0.01, min(1.0, new_w))
    new_h = max(0.01, min(1.0, new_h))
    new_x_c = max(new_w/2, min(1.0 - new_w/2, new_x_c))
    new_y_c = max(new_h/2, min(1.0 - new_h/2, new_y_c))
    
    return new_x_c, new_y_c, new_w, new_h


def build_transforms(img_size: int = 640, augment: bool = True) -> Callable:
    """
    Build image transformation pipeline.
    
    Args:
        img_size: Target square size
        augment: Enable augmentation
        
    Returns:
        Transform function
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def transform(image: Union[str, Path, np.ndarray], aug_params: dict = None) -> torch.Tensor:
        """
        Transform image to normalized tensor.
        
        Args:
            image: Input image (path or numpy array)
            aug_params: Augmentation parameters
            
        Returns:
            Normalized tensor (3, img_size, img_size)
        """
        # Load + convert BGR→RGB + resize
        img = _load_image(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        pil = T.functional.to_pil_image(img)
        tensor = T.functional.to_tensor(pil)
        
        if augment and aug_params is not None:
            # Geometric augmentation
            if abs(aug_params['angle']) > 0.1:
                tensor = TF.rotate(tensor, aug_params['angle'], interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            
            if aug_params['flip_h']:
                tensor = TF.hflip(tensor)
            
            if aug_params['flip_v']:
                tensor = TF.vflip(tensor)
            
            # Color augmentation
            pil = T.functional.to_pil_image(tensor)
            pil = TF.adjust_brightness(pil, aug_params['brightness'])
            pil = TF.adjust_contrast(pil, aug_params['contrast'])
            pil = TF.adjust_saturation(pil, aug_params['saturation'])
            tensor = T.functional.to_tensor(pil)
        
        # Normalize
        return normalize(tensor)
    
    return transform
