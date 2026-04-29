import torch
import torchvision.transforms as T
import argparse
from PIL import Image
from transformers import AutoTokenizer

# Import our model
from models.generator import GeoNarrator

def load_image(image_path, image_size=256):
    """Loads and preprocesses a single image for inference."""
    # Standard ImageNet normalization (must match training exactly)
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) # Add batch dimension: (1, C, H, W)
    return image

def main(image_path_A, image_path_B, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")
    
    # 1. Load Tokenizer (Must match what was used in training)
    # Using the standard BERT tokenizer as a baseline
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # 2. Load Configuration (Mocked here, but usually loaded from your default.yaml)
    config = {
        'model': {
            'encoder': {'type': 'swin_base_patch4_window7_224', 'pretrained': False, 'freeze_backbone': False},
            'fusion': {'type': 'cross_attention', 'embed_dim': 768},
            'decoder': {'vocab_size': 30522, 'hidden_dim': 768, 'num_heads': 8, 'num_layers': 4}
        }
    }
    
    # 3. Initialize Model
    model = GeoNarrator(config).to(device)
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load weights. Running with untrained weights for demonstration. Error: {e}")
    
    # Set to evaluation mode! Crucial for disabling dropout
    model.eval()
    
    # 4. Load and Preprocess Images
    img_A = load_image(image_path_A).to(device)
    img_B = load_image(image_path_B).to(device)
    
    # 5. Generate Text
    start_token = tokenizer.cls_token_id  # <SOS> token
    eos_token = tokenizer.sep_token_id    # <EOS> token
    
    print("Generating narrative...")
    with torch.no_grad(): # Turn off gradients to save memory and speed up inference
        output_tokens = model.generate(
            img_A, 
            img_B, 
            start_token_id=start_token, 
            eos_token_id=eos_token, 
            max_length=50
        )
    
    # 6. Decode output tokens back into an English sentence
    predicted_sentence = tokenizer.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("🌍 GEONARRATOR OUTPUT")
    print("="*50)
    print(f"Narrative: {predicted_sentence.capitalize()}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on GeoNarrator")
    parser.add_argument("--image_A", type=str, required=True, help="Path to the 'Before' satellite image")
    parser.add_argument("--image_B", type=str, required=True, help="Path to the 'After' satellite image")
    parser.add_argument("--weights", type=str, default="weights/best_model.pth", help="Path to trained .pth file")
    
    args = parser.parse_args()
    main(args.image_A, args.image_B, args.weights)