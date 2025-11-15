"""Decoder with template and deformable attention."""

import torch
import torch.nn as nn

from .deformable_attention import MultiScaleDeformableAttention


class DeformableDecoderLayer(nn.Module):
    """
    Single decoder layer with 4 stages:
    1. Self-attention (query-to-query)
    2. Template cross-attention (query-to-template tokens)
    3. Deformable search cross-attention (query-to-search features)
    4. Feed-forward network
    """
    
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1,
                 num_levels=3, num_points=4):
        super().__init__()
        
        # 1. Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Template cross-attention
        self.template_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 3. Deformable search cross-attention
        self.deformable_attn = MultiScaleDeformableAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_levels=num_levels,
            num_points=num_points
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 4. Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(d_model)
    
    def forward(self, query, template_tokens, search_features, spatial_shapes, reference_points):
        """
        Args:
            query: (B, num_queries, d_model)
            template_tokens: (B, N*3, d_model) - e.g., (B, 9, 256)
            search_features: List of 3 feature maps
            spatial_shapes: List of 3 (H, W) tuples
            reference_points: (B, num_queries, 2) - normalized coordinates
        
        Returns:
            query: (B, num_queries, d_model)
        """
        # 1. Self-attention
        q2 = self.self_attn(query, query, query)[0]
        query = query + self.dropout1(q2)
        query = self.norm1(query)
        
        # 2. Template cross-attention
        q2 = self.template_attn(query, template_tokens, template_tokens)[0]
        query = query + self.dropout2(q2)
        query = self.norm2(query)
        
        # 3. Deformable search cross-attention
        q2 = self.deformable_attn(query, reference_points, search_features, spatial_shapes)
        query = query + self.dropout3(q2)
        query = self.norm3(query)
        
        # 4. Feed-forward
        q2 = self.ffn(query)
        query = query + q2
        query = self.norm4(query)
        
        return query


class DeformableDecoder(nn.Module):
    """Stack of decoder layers."""
    
    def __init__(self, num_layers=6, d_model=256, nhead=8, dim_feedforward=1024,
                 dropout=0.1, num_levels=3, num_points=4):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DeformableDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_levels=num_levels,
                num_points=num_points
            )
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(self, query, template_tokens, search_features, spatial_shapes, reference_points):
        """
        Args:
            query: (B, num_queries, d_model)
            template_tokens: (B, N*3, d_model)
            search_features: List of 3 feature maps
            spatial_shapes: List of 3 (H, W) tuples
            reference_points: (B, num_queries, 2)
        
        Returns:
            query: (B, num_queries, d_model)
        """
        for layer in self.layers:
            query = layer(query, template_tokens, search_features, spatial_shapes, reference_points)
        
        return query


def build_decoder(num_layers=6, d_model=256, nhead=8, dim_feedforward=1024,
                  dropout=0.1, num_levels=3, num_points=4):
    """Factory function."""
    return DeformableDecoder(
        num_layers=num_layers,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_levels=num_levels,
        num_points=num_points
    )
