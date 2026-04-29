import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """
    Fuses the 'Before' (feat_A) and 'After' (feat_B) spatial feature maps.
    Supports subtraction, concatenation, and cross-attention.
    """
    def __init__(self, fusion_type='cross_attention', embed_dim=768):
        super().__init__()
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim
        
        if fusion_type == 'concatenation':
            # Project the stacked embeddings (2 * embed_dim) back down to embed_dim
            self.proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
            
        elif fusion_type == 'cross_attention':
            # PyTorch's MultiheadAttention expects (Batch, Sequence, Features) when batch_first=True
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
            
            # Standard Transformer normalization and feed-forward layers
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )

    def forward(self, feat_A, feat_B):
        """
        Args:
            feat_A: Tensor (B, C, H, W) - The "Before" features
            feat_B: Tensor (B, C, H, W) - The "After" features
        Returns:
            fused: Tensor (B, C, H, W) - The combined change embedding
        """
        B, C, H, W = feat_A.shape
        
        if self.fusion_type == 'subtraction':
            # The simplest delta: absolute mathematical difference
            return torch.abs(feat_B - feat_A)
            
        elif self.fusion_type == 'concatenation':
            # Stack along the channel dimension: (B, 2C, H, W)
            stacked = torch.cat([feat_A, feat_B], dim=1)
            # Use 1x1 convolution to compress back to (B, C, H, W)
            return self.proj(stacked)
            
        elif self.fusion_type == 'cross_attention':
            # 1. Flatten spatial grid into sequences: (B, C, H, W) -> (B, H*W, C)
            feat_A_seq = feat_A.flatten(2).transpose(1, 2)
            feat_B_seq = feat_B.flatten(2).transpose(1, 2)
            
            # 2. Cross-Attention: The "After" image queries the "Before" image
            # Q = After (feat_B), K = Before (feat_A), V = Before (feat_A)
            attn_output, _ = self.cross_attn(
                query=feat_B_seq, 
                key=feat_A_seq, 
                value=feat_A_seq
            )
            
            # 3. Add & Norm (Residual connection helps gradients flow)
            x = self.norm1(feat_B_seq + attn_output)
            
            # 4. Feed-Forward & Norm
            fused_seq = self.norm2(x + self.ffn(x))
            
            # 5. Reshape sequence back into a 2D spatial feature map
            fused_map = fused_seq.transpose(1, 2).view(B, C, H, W)
            return fused_map
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

# --- Quick Test Block ---
if __name__ == "__main__":
    print("Testing Feature Fusion Module...")
    # Dummy tensors representing Swin Transformer output maps (Batch=2, Channels=768, 8x8 grid)
    dummy_A = torch.randn(2, 768, 8, 8)
    dummy_B = torch.randn(2, 768, 8, 8)
    
    fusion_module = FeatureFusion(fusion_type='cross_attention', embed_dim=768)
    fused_output = fusion_module(dummy_A, dummy_B)
    
    print(f"Fused Feature Map Shape: {fused_output.shape}") # Should be [2, 768, 8, 8]