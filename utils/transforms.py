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
from PIL import ImageFilter


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
    scale: float = 1.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    Transform bounding box coordinates according to geometric augmentations.
    
    Args:
        x_c, y_c, w, h: Normalized bbox coordinates (0-1) in format (center_x, center_y, width, height)
        img_size: Image size (assumed square)
        scale: Scale factor (applied first)
        translate_x: Translation in x direction (normalized, applied after scale)
        translate_y: Translation in y direction (normalized, applied after scale)
        angle: Rotation angle in degrees (applied after scale/translate)
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
    
    # 1. Scale (zoom in/out around center)
    if abs(scale - 1.0) > 1e-6:
        corners = (corners - center) * scale + center
        # After scale, we need to handle cropping - but for simplicity,
        # we'll just scale the bbox (the image will be cropped/padded in transform)
    
    # 2. Translate
    if abs(translate_x) > 1e-6 or abs(translate_y) > 1e-6:
        translate_x_px = translate_x * img_size
        translate_y_px = translate_y * img_size
        corners[:, 0] += translate_x_px
        corners[:, 1] += translate_y_px
    
    # 3. Rotation
    if abs(angle) > 1e-6:
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        corners = (corners - center) @ rot_matrix.T + center
    
    # 4. Horizontal flip
    if flip_h:
        corners[:, 0] = img_size - corners[:, 0]
    
    # 5. Vertical flip
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
            aug_params: Augmentation parameters (None = no augment)
            
        Returns:
            Normalized tensor (3, img_size, img_size)
        """
        # Load + convert BGR→RGB
        img = _load_image(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation if enabled and params provided
        if augment and aug_params is not None:
            # 1. Scale (zoom in/out) - apply before resize
            scale = aug_params.get('scale', 1.0)
            if abs(scale - 1.0) > 1e-6:
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Crop/pad to original size (center crop)
                if scale > 1.0:
                    # Crop center
                    start_y = (new_h - h) // 2
                    start_x = (new_w - w) // 2
                    img = img[start_y:start_y+h, start_x:start_x+w]
                else:
                    # Pad with zeros
                    pad_y = (h - new_h) // 2
                    pad_x = (w - new_w) // 2
                    img = cv2.copyMakeBorder(img, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            # 2. Translate (shift) - apply before resize
            translate_x = aug_params.get('translate', 0.0)
            translate_y = aug_params.get('translate', 0.0)  # Same value for both (can be improved)
            if abs(translate_x) > 1e-6 or abs(translate_y) > 1e-6:
                h, w = img.shape[:2]
                tx = int(translate_x * w)
                ty = int(translate_y * h)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0])
            
            # Resize to target size
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            pil = T.functional.to_pil_image(img)
            
            # 3. Rotation
            if abs(aug_params.get('angle', 0)) > 0.1:
                pil = TF.rotate(pil, aug_params['angle'], interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            
            # 4. Horizontal flip
            if aug_params['flip_h']:
                pil = TF.hflip(pil)
            
            # 5. Color augmentation
            pil = TF.adjust_brightness(pil, aug_params['brightness'])
            pil = TF.adjust_contrast(pil, aug_params['contrast'])
            pil = TF.adjust_saturation(pil, aug_params['saturation'])
            pil = TF.adjust_hue(pil, aug_params['hue'])
            
            # 6. Blur augmentation
            if aug_params.get('blur', False):
                radius = random.uniform(0.5, 2.0)
                pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
            
            # Convert to tensor for remaining augmentations
            tensor = T.functional.to_tensor(pil)
            
            # 7. Gaussian noise
            if aug_params.get('noise', False):
                noise = torch.randn_like(tensor) * 0.05
                tensor = torch.clamp(tensor + noise, 0, 1)
            
            # 8. Cutout
            if aug_params.get('cutout', False):
                _, h, w = tensor.shape
                cutout_size = int(random.uniform(0.02, 0.05) * min(h, w))
                y = random.randint(0, h - cutout_size)
                x = random.randint(0, w - cutout_size)
                tensor[:, y:y+cutout_size, x:x+cutout_size] = 0
        else:
            # No augmentation - just resize and convert to tensor
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            pil = T.functional.to_pil_image(img)
            tensor = T.functional.to_tensor(pil)
        
        # Normalize
        return normalize(tensor)
    
    return transform

