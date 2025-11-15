"""Positional encoding for features."""

import math
import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """
    2D sine-cosine positional encoding.
    """
    
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        
        Returns:
            pos: (B, C, H, W) positional encoding
        """
        B, C, H, W = x.shape
        
        # Create coordinate grids
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device)
        
        if self.normalize:
            y_embed = y_embed / (H - 1) * self.scale
            x_embed = x_embed / (W - 1) * self.scale
        
        # Compute sine and cosine embeddings
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        
        # Stack sine and cosine
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        
        # Combine x and y: (H, num_pos_feats), (W, num_pos_feats)
        pos = torch.cat((pos_y[:, None, :].expand(H, W, self.num_pos_feats),
                         pos_x[None, :, :].expand(H, W, self.num_pos_feats)), dim=-1)
        
        # (H, W, 2*num_pos_feats) -> (1, 2*num_pos_feats, H, W)
        pos = pos.permute(2, 0, 1).unsqueeze(0)
        
        # Expand to batch size
        pos = pos.expand(B, -1, -1, -1)
        
        return pos


def build_position_encoding(hidden_dim=256, position_embedding='sine'):
    """Factory function."""
    assert position_embedding in ['sine'], f"Unknown position_embedding: {position_embedding}"
    
    num_pos_feats = hidden_dim // 2
    
    return PositionEmbeddingSine(num_pos_feats, normalize=True)
