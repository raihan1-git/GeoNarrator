import torch
import torch.nn as nn
import timm

class SiameseEncoder(nn.Module):
    """
    A Siamese Vision Encoder that processes two temporal images (Before and After)
    through the same shared weights to extract hierarchical feature maps.
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # We use timm to load the transformer.
        # Setting num_classes=0 and global_pool='' strips away the classification head.
        # This forces the model to return the raw 2D spatial feature maps, 
        # which we need to figure out exactly *where* the change happened.
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0, 
            global_pool=''
        )
        
        # Optional: Freeze the backbone to save VRAM and speed up initial training,
        # forcing only the Fusion and Language Decoder modules to learn.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, image_A, image_B):
        """
        Args:
            image_A: Tensor of shape (B, C, H, W) - The "Before" image
            image_B: Tensor of shape (B, C, H, W) - The "After" image
            
        Returns:
            Tuple of feature tensors (features_A, features_B)
        """
        # The Siamese mechanism: Both images pass through the EXACT SAME network.
        # This ensures the mathematical baseline for comparison is identical.
        features_A = self.backbone(image_A)
        features_B = self.backbone(image_B)
        
        # Transformer outputs can vary depending on the specific ViT/Swin architecture.
        # If the output is a flattened sequence (B, L, C), we reshape it back to a 
        # spatial grid (B, C, H, W) so the fusion module can compute spatial differences.
        if features_A.dim() == 3:
            B, L, C = features_A.shape
            H = W = int(L ** 0.5)
            features_A = features_A.view(B, H, W, C).permute(0, 3, 1, 2)
            features_B = features_B.view(B, H, W, C).permute(0, 3, 1, 2)
            
        return features_A, features_B

# --- Quick Test Block ---
if __name__ == "__main__":
    print("Initializing Swin Transformer Siamese Encoder...")
    model = SiameseEncoder(model_name='swin_tiny_patch4_window7_224')
    
    # Create two dummy images representing a batch of 2 satellite image pairs
    dummy_A = torch.randn(2, 3, 224, 224)
    dummy_B = torch.randn(2, 3, 224, 224)
    
    feat_A, feat_B = model(dummy_A, dummy_B)
    print(f"Feature A shape: {feat_A.shape}")
    print(f"Feature B shape: {feat_B.shape}")