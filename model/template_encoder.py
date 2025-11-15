"""Template encoder with global pooling for multi-scale features."""

import torch
import torch.nn as nn


class TemplateEncoder(nn.Module):
    """
    Encode templates to compact token representation.
    
    Input: N templates, each with multi-scale features {s2, s3, s4}
    Output: (B, N*3, hidden_dim) tokens - 3 scales per template
    
    For 3 templates: (B, 9, 256) - 9 compact tokens
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
    def forward(self, template_features_list):
        """
        Args:
            template_features_list: List of N dicts, each containing:
                - 's2': (B, 256, H2, W2)
                - 's3': (B, 256, H3, W3)
                - 's4': (B, 256, H4, W4)
        
        Returns:
            tokens: (B, N*3, 256) - Global pooled tokens
        """
        batch_size = template_features_list[0]['s2'].shape[0]
        num_templates = len(template_features_list)
        
        all_tokens = []
        
        for template_feats in template_features_list:
            # Global average pool each scale
            s2_token = torch.mean(template_feats['s2'], dim=[2, 3])  # (B, 256)
            s3_token = torch.mean(template_feats['s3'], dim=[2, 3])  # (B, 256)
            s4_token = torch.mean(template_feats['s4'], dim=[2, 3])  # (B, 256)
            
            # Stack: (B, 3, 256)
            template_tokens = torch.stack([s2_token, s3_token, s4_token], dim=1)
            all_tokens.append(template_tokens)
        
        # Concatenate all templates: (B, N*3, 256)
        tokens = torch.cat(all_tokens, dim=1)
        
        return tokens


def build_template_encoder(hidden_dim=256):
    """Factory function."""
    return TemplateEncoder(hidden_dim=hidden_dim)
