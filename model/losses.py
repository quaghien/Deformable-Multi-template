"""Loss functions for Deformable RefDet (adapted from CA-DETR)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

from .matcher import HungarianMatcher, generalized_box_iou


class SetCriterion(nn.Module):
    """
    Loss computation for DETR-style models.
    
    Combines:
    - Focal Loss for classification
    - L1 Loss for bbox regression
    - GIoU Loss for bbox quality
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        matcher: HungarianMatcher = None,
        weight_dict: Dict[str, float] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            num_classes: Number of object classes (1 for binary detection)
            matcher: Hungarian matcher for matching predictions to targets
            weight_dict: Dict of loss weights {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher if matcher is not None else HungarianMatcher()
        
        if weight_dict is None:
            weight_dict = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
        self.weight_dict = weight_dict
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def loss_labels(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: int) -> Dict:
        """
        Compute focal loss for classification.
        
        Args:
            outputs: Model outputs with 'pred_logits' (B, num_queries, 1)
            targets: List of target dicts with 'labels'
            indices: Matching indices from Hungarian matcher
            num_boxes: Number of boxes for normalization
            
        Returns:
            losses: Dict with 'loss_ce'
        """
        pred_logits = outputs['pred_logits']  # (B, num_queries, 1)
        
        # Create target tensor
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(pred_logits.shape[:2], dtype=torch.float32, device=pred_logits.device)
        target_classes[idx] = target_classes_o.float()
        
        # Focal loss
        pred_probs = pred_logits.sigmoid().squeeze(-1)  # (B, num_queries)
        target_classes = target_classes  # (B, num_queries)
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(-1), target_classes, reduction='none'
        )
        
        p_t = pred_probs * target_classes + (1 - pred_probs) * (1 - target_classes)
        alpha_t = self.focal_alpha * target_classes + (1 - self.focal_alpha) * (1 - target_classes)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        
        # Normalize by number of queries
        loss_ce = (focal_weight * ce_loss).sum() / max(pred_logits.shape[0] * pred_logits.shape[1], 1)
        
        losses = {'loss_ce': loss_ce}
        return losses
    
    def loss_boxes(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: int) -> Dict:
        """
        Compute L1 and GIoU losses for bbox regression.
        
        Args:
            outputs: Model outputs with 'pred_boxes' (B, num_queries, 4)
            targets: List of target dicts with 'boxes'
            indices: Matching indices from Hungarian matcher
            num_boxes: Number of boxes for normalization
            
        Returns:
            losses: Dict with 'loss_bbox' and 'loss_giou'
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # (num_matched, 4)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        
        # GIoU loss
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        loss_giou = loss_giou.sum() / num_boxes
        
        losses = {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
        return losses
    
    def _get_src_permutation_idx(self, indices: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get permutation indices for matched predictions."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get permutation indices for matched targets."""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict:
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs with 'pred_logits' and 'pred_boxes'
            targets: List of target dicts with 'labels' and 'boxes'
            
        Returns:
            losses: Dict of all losses
        """
        # Match predictions to targets
        indices = self.matcher(outputs, targets)
        
        # Number of boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        
        # Compute losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        
        # Apply weights
        weighted_losses = {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}
        
        return weighted_losses


def build_criterion(
    num_classes: int = 1,
    weight_dict: Dict[str, float] = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
) -> SetCriterion:
    """
    Build loss criterion.
    
    Args:
        num_classes: Number of classes (1 for binary)
        weight_dict: Loss weights
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
        
    Returns:
        criterion: SetCriterion instance
    """
    if weight_dict is None:
        weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0
        }
    
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0
    )
    
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )
    
    return criterion
