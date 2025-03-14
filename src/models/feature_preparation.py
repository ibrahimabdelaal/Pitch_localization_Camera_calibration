import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSformerFeaturePrep(nn.Module):
    def __init__(
        self,
        input_dims={'early': 352, 'middle': 704, 'deep': 1408},
        output_dim=768,  # TimeSformer expected dimension
        target_resolution='early'  # Use early feature resolution (H/4)
    ):
        super().__init__()
        
        # 1. Project each feature level to TimeSformer dimension
        self.projections = nn.ModuleDict({
            name: nn.Conv2d(dim, output_dim, kernel_size=1)
            for name, dim in input_dims.items()
        })
        
        # 2. Simple but effective fusion
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, features, num_frames):
        """
        Args:
            features: Dict of features from ConvNeXt
                'early':  [B*T, 352, H/4, W/4]
                'middle': [B*T, 704, H/8, W/8]
                'deep':   [B*T, 1408, H/16, W/16]
            num_frames: Number of frames in sequence (T)
            
        Returns:
            tensor: Shape [B, T, 768, H/4, W/4] ready for TimeSformer
        """
        B = features['early'].shape[0] // num_frames
        target_size = features['early'].shape[2:]  # H/4, W/4
        
        # Process each feature level
        processed = {}
        for level, feat in features.items():
            # 1. Project to TimeSformer dimension
            x = self.projections[level](feat)
            
            # 2. Resize to target resolution if needed
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            
            processed[level] = x
        
        # 3. Weighted fusion
        fused = sum(w * f for w, f in zip(
            self.fusion_weights, 
            processed.values()
        ))
        
        # 4. Reshape to TimeSformer format [B, T, C, H, W]
        fused = fused.reshape(B, num_frames, *fused.shape[1:])
        
        # 5. Apply normalization
        fused = self.norm(fused.permute(0,1,3,4,2)).permute(0,1,4,2,3)
        
        return fused

def test_feature_prep():
    """Test the feature preparation module."""
    # Create dummy features
    B, T = 2, 4  # batch_size, num_frames
    H, W = 56, 56  # H/4, W/4 resolution
    
    features = {
        'early':  torch.randn(B*T, 352, H, W),
        'middle': torch.randn(B*T, 704, H//2, W//2),
        'deep':   torch.randn(B*T, 1408, H//4, W//4)
    }
    
    # Initialize module
    prep = TimeSformerFeaturePrep()
    
    # Process features
    output = prep(features, T)
    
    # Print shapes
    print(f"\nInput shapes:")
    for k, v in features.items():
        print(f"{k}: {v.shape}")
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [B, T, C, H, W] = [{B}, {T}, 768, {H}, {W}]")

if __name__ == "__main__":
    test_feature_prep() 