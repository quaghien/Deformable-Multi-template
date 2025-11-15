"""Swin Transformer backbone for multi-scale feature extraction."""

import torch
import torch.nn as nn
import timm


class SwinBackbone(nn.Module):
    """
    Swin-Tiny backbone with multi-scale output.
    
    Output stages: S2 (28×28, C=192), S3 (14×14, C=384), S4 (7×7, C=768)
    All projected to out_channels=256 via 1×1 conv.
    """
    
    def __init__(self, model_name='swin_tiny_patch4_window7_224', 
                 pretrained=True, out_channels=256):
        super().__init__()
        
        # Load pretrained Swin-Tiny from timm
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,  # Return multi-scale features
            out_indices=(1, 2, 3),  # S2, S3, S4 (skip S1 at 56×56)
            strict_img_size=False,  # Allow arbitrary input sizes
        )
        
        # Get feature channels from each stage
        # Swin-Tiny: S2=192, S3=384, S4=768
        self.feature_channels = self.swin.feature_info.channels()  # [192, 384, 768]
        
        # Project each stage to unified dimension
        self.proj_s2 = nn.Conv2d(self.feature_channels[0], out_channels, 1)
        self.proj_s3 = nn.Conv2d(self.feature_channels[1], out_channels, 1)
        self.proj_s4 = nn.Conv2d(self.feature_channels[2], out_channels, 1)
        
        self.out_channels = out_channels
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - Input images (H=W=640)
            
        Returns:
            dict: {'s2': (B,256,H/8,W/8), 's3': (B,256,H/16,W/16), 's4': (B,256,H/32,W/32)}
        """
        # Extract multi-scale features (NHWC format from Swin)
        features = self.swin(x)  # List of 3 tensors in NHWC format
        
        # Convert NHWC -> NCHW
        s2_feat = features[0].permute(0, 3, 1, 2)  # (B, 192, H/8, W/8)
        s3_feat = features[1].permute(0, 3, 1, 2)  # (B, 384, H/16, W/16)
        s4_feat = features[2].permute(0, 3, 1, 2)  # (B, 768, H/32, W/32)
        
        # Project to unified dimension
        s2 = self.proj_s2(s2_feat)  # (B, 256, H/8, W/8)
        s3 = self.proj_s3(s3_feat)  # (B, 256, H/16, W/16)
        s4 = self.proj_s4(s4_feat)  # (B, 256, H/32, W/32)
        
        return {'s2': s2, 's3': s3, 's4': s4}


def build_swin_backbone(pretrained=True, out_channels=256):
    """Build Swin-Tiny backbone."""
    return SwinBackbone(
        model_name='swin_tiny_patch4_window7_224',
        pretrained=pretrained,
        out_channels=out_channels
    )


if __name__ == "__main__":
    # Test backbone
    model = build_swin_backbone(pretrained=False)
    x = torch.randn(2, 3, 640, 640)
    
    with torch.no_grad():
        out = model(x)
    
    print("Swin-Tiny Backbone Test:")
    for key, val in out.items():
        print(f"  {key}: {val.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params/1e6:.2f}M")
