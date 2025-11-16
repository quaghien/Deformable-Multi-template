"""Multi-scale deformable attention implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDeformableAttention(nn.Module):
    """
    Deformable attention over multi-scale features.
    
    Samples num_points from each scale level at learned offset positions.
    Total sampling points: num_levels * num_points (e.g., 3 * 4 = 12)
    """
    
    def __init__(self, embed_dim=256, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_levels * num_points
        
        # Offset prediction: predict 2D offsets for each point
        self.offset_proj = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        
        # Attention weights
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        
        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize offset to sample around reference point
        nn.init.constant_(self.offset_proj.weight, 0.0)
        nn.init.constant_(self.offset_proj.bias, 0.0)
        
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
    
    def forward(self, query, reference_points, value_features, spatial_shapes):
        """
        Args:
            query: (B, num_queries, embed_dim)
            reference_points: (B, num_queries, 2) - normalized [0, 1] coordinates
            value_features: List of 3 feature maps
                - (B, embed_dim, H_s2, W_s2)
                - (B, embed_dim, H_s3, W_s3)
                - (B, embed_dim, H_s4, W_s4)
            spatial_shapes: List of 3 tuples (H, W)
        
        Returns:
            output: (B, num_queries, embed_dim)
        """
        B, num_queries, embed_dim = query.shape
        
        # Predict offsets: (B, num_queries, num_heads, num_levels, num_points, 2)
        offsets = self.offset_proj(query)
        offsets = offsets.view(B, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
        offsets = torch.tanh(offsets) * 0.5  # Scale to [-0.5, 0.5]
        
        # Predict attention weights: (B, num_queries, num_heads, num_levels, num_points)
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(B, num_queries, self.num_heads, self.num_levels, self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Expand reference points: (B, num_queries, 1, num_levels, 1, 2)
        ref_points = reference_points.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        # Add offsets to reference points
        sampling_locations = ref_points + offsets  # (B, num_queries, num_heads, num_levels, num_points, 2)
        
        # Sample from each level
        sampled_features = []
        
        for lvl_idx, value_feat in enumerate(value_features):
            # value_feat: (B, embed_dim, H, W)
            B, C, H, W = value_feat.shape
            
            # Get sampling locations for this level: (B, num_queries, num_heads, num_points, 2)
            lvl_sampling_loc = sampling_locations[:, :, :, lvl_idx, :, :]
            
            # Reshape for grid_sample: (B * num_queries * num_heads, 1, num_points, 2)
            lvl_sampling_loc = lvl_sampling_loc.reshape(B * num_queries * self.num_heads, 1, self.num_points, 2)
            
            # Convert from [0, 1] to [-1, 1] for grid_sample
            lvl_sampling_loc = lvl_sampling_loc * 2.0 - 1.0
            
            # Split embed_dim into num_heads * head_dim BEFORE sampling
            # (B, embed_dim, H, W) -> (B, num_heads, head_dim, H, W)
            value_feat_split = value_feat.view(B, self.num_heads, self.head_dim, H, W)
            
            # Expand for each query: (B, num_heads, head_dim, H, W) -> (B * num_queries, num_heads, head_dim, H, W)
            value_feat_expanded = value_feat_split.unsqueeze(1).expand(B, num_queries, self.num_heads, self.head_dim, H, W)
            value_feat_expanded = value_feat_expanded.reshape(B * num_queries, self.num_heads, self.head_dim, H, W)
            
            # Reshape for grid_sample: (B * num_queries * num_heads, head_dim, H, W)
            value_feat_expanded = value_feat_expanded.reshape(B * num_queries * self.num_heads, self.head_dim, H, W)
            
            # Sample: (B * num_queries * num_heads, head_dim, 1, num_points)
            sampled = F.grid_sample(
                value_feat_expanded,
                lvl_sampling_loc,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )
            
            # Reshape: (B, num_queries, num_heads, head_dim, num_points)
            sampled = sampled.reshape(B, num_queries, self.num_heads, self.head_dim, self.num_points)
            sampled_features.append(sampled)
        
        # Stack all levels: (B, num_queries, num_heads, head_dim, num_levels, num_points)
        sampled_features = torch.stack(sampled_features, dim=4)
        
        # Apply attention weights: (B, num_queries, num_heads, num_levels, num_points)
        # Unsqueeze at dim=3 (after head_dim) to broadcast correctly
        attention_weights = attention_weights.unsqueeze(3)  # (B, num_queries, num_heads, 1, num_levels, num_points)
        
        # Weighted sum: (B, num_queries, num_heads, head_dim)
        output = (sampled_features * attention_weights).sum(dim=[4, 5])
        
        # Concatenate heads: (B, num_queries, num_heads, head_dim) -> (B, num_queries, embed_dim)
        output = output.reshape(B, num_queries, self.embed_dim)
        
        # Output projection
        output = self.output_proj(output)
        
        return output


def build_deformable_attention(embed_dim=256, num_heads=8, num_levels=3, num_points=4):
    """Factory function."""
    return MultiScaleDeformableAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points
    )
