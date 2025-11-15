"""Main Deformable Reference Detection model."""

import torch
import torch.nn as nn

from .swin_backbone import build_swin_backbone
from .template_encoder import build_template_encoder
from .decoder import build_decoder
from .positional_encoding import build_position_encoding


class DeformableRefDet(nn.Module):
    """
    Deformable Reference Detection with Swin-Tiny backbone.
    
    Architecture:
    1. Swin-Tiny backbone extracts multi-scale features (S2, S3, S4)
    2. Template encoder pools 3 templates → 9 compact tokens
    3. Query embeddings initialized with template conditioning
    4. Deformable decoder attends to template tokens + search features
    5. Classification + bbox regression heads
    """
    
    def __init__(self, num_classes=1, num_queries=5, hidden_dim=256,
                 num_decoder_layers=6, num_heads=8, dim_feedforward=1024,
                 dropout=0.1, num_levels=3, num_points=4, pretrained_backbone=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        
        # Backbone
        self.backbone = build_swin_backbone(pretrained=pretrained_backbone)
        
        # Template encoder
        self.template_encoder = build_template_encoder(hidden_dim=hidden_dim)
        
        # Positional encoding
        self.position_encoding = build_position_encoding(hidden_dim=hidden_dim)
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Template conditioning for query initialization
        # 9 template tokens → condition query embeddings
        self.query_init = nn.Linear(9 * hidden_dim, num_queries * hidden_dim)
        
        # Reference point initialization
        self.reference_points = nn.Linear(hidden_dim, 2)
        
        # Decoder
        self.decoder = build_decoder(
            num_layers=num_decoder_layers,
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_levels=num_levels,
            num_points=num_points
        )
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_init.weight)
        nn.init.constant_(self.query_init.bias, 0.0)
        
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.5)  # Initialize to center
        
        nn.init.xavier_uniform_(self.class_embed.weight)
        nn.init.constant_(self.class_embed.bias, 0.0)
        
        for layer in self.bbox_embed:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, templates, search):
        """
        Args:
            templates: (B, 3, 3, H, W) - 3 templates, 3 channels
            search: (B, 3, H, W) - search image
        
        Returns:
            pred_logits: (B, num_queries, num_classes)
            pred_boxes: (B, num_queries, 4) - normalized [cx, cy, w, h]
        """
        B = search.shape[0]
        
        # Extract template features
        template_features_list = []
        for i in range(templates.shape[1]):  # 3 templates
            template_i = templates[:, i]  # (B, 3, H, W)
            feats_i = self.backbone(template_i)
            template_features_list.append(feats_i)
        
        # Extract search features
        search_features = self.backbone(search)
        
        # Encode templates to compact tokens: (B, 9, 256)
        template_tokens = self.template_encoder(template_features_list)
        
        # Initialize queries with template conditioning
        # Flatten template tokens: (B, 9*256)
        template_flat = template_tokens.flatten(1)
        
        # Predict query initialization: (B, num_queries * hidden_dim)
        query_init = self.query_init(template_flat)
        query_init = query_init.view(B, self.num_queries, self.hidden_dim)
        
        # Add learnable query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        query = query_init + query_embed
        
        # Initialize reference points: (B, num_queries, 2)
        reference_points = self.reference_points(query).sigmoid()
        
        # Prepare multi-scale search features as list
        search_feature_list = [
            search_features['s2'],
            search_features['s3'],
            search_features['s4']
        ]
        
        # Add positional encoding to search features
        for i in range(len(search_feature_list)):
            pos_enc = self.position_encoding(search_feature_list[i])
            search_feature_list[i] = search_feature_list[i] + pos_enc
        
        # Spatial shapes
        spatial_shapes = [
            (search_features['s2'].shape[2], search_features['s2'].shape[3]),
            (search_features['s3'].shape[2], search_features['s3'].shape[3]),
            (search_features['s4'].shape[2], search_features['s4'].shape[3])
        ]
        
        # Decode
        query = self.decoder(
            query=query,
            template_tokens=template_tokens,
            search_features=search_feature_list,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points
        )
        
        # Predict classification and bboxes
        pred_logits = self.class_embed(query)  # (B, num_queries, num_classes)
        pred_boxes = self.bbox_embed(query).sigmoid()  # (B, num_queries, 4)
        
        return pred_logits, pred_boxes


def build_model(num_classes=1, num_queries=5, hidden_dim=256,
                num_decoder_layers=6, num_heads=8, dim_feedforward=1024,
                dropout=0.1, num_levels=3, num_points=4, pretrained_backbone=True):
    """Factory function."""
    return DeformableRefDet(
        num_classes=num_classes,
        num_queries=num_queries,
        hidden_dim=hidden_dim,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_levels=num_levels,
        num_points=num_points,
        pretrained_backbone=pretrained_backbone
    )
