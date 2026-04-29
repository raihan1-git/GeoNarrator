import torchvision.transforms as T

def get_transforms(image_size=256, is_train=True):
    """
    Returns the transformation pipeline for the images.
    Swin Transformers (pretrained on ImageNet) expect very specific normalization.
    """
    # Standard ImageNet statistics expected by Swin/ViT
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        # Training pipeline includes random augmentations to prevent overfitting
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        # Validation/Testing pipeline strictly resizes and normalizes
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])