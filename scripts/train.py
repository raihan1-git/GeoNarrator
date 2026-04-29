import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom modules
from models.generator import GeoNarrator
from data.dataset import GeoNarratorDataset
from data.transforms import get_transforms

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    # Progress bar for the terminal
    loop = tqdm(dataloader, leave=True)
    
    for batch in loop:
        # Move data to GPU
        img_A = batch['image_A'].to(device)
        img_B = batch['image_B'].to(device)
        tokens = batch['input_ids'].to(device)
        
        # Create a padding mask for the transformer
        # (Assuming 0 is the ID for <PAD> in your tokenizer)
        pad_token_id = 0 
        padding_mask = (tokens == pad_token_id)
        
        # 1. Zero the gradients
        optimizer.zero_grad()
        
        # 2. Forward Pass: We pass tokens[:, :-1] (all words except the last) 
        # as the input to predict tokens[:, 1:] (the next words)
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        padding_mask = padding_mask[:, :-1]
        
        logits = model(img_A, img_B, input_tokens, padding_mask)
        
        # 3. Calculate Loss
        # PyTorch CrossEntropy expects (Batch*SeqLen, VocabSize) and (Batch*SeqLen)
        logits = logits.reshape(-1, logits.shape[-1]) 
        target_tokens = target_tokens.reshape(-1)
        
        loss = criterion(logits, target_tokens)
        
        # 4. Backward Pass (Calculate Gradients)
        loss.backward()
        
        # 5. Optimizer Step (Update Weights)
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_description(f"Epoch Loss: {loss.item():.4f}")
        
    return total_loss / len(dataloader)

if __name__ == "__main__":
    # --- Configuration Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Mock config (In reality, you would load this from default.yaml)
    mock_config = {
        'model': {
            'encoder': {'type': 'swin_tiny_patch4_window7_224', 'pretrained': True, 'freeze_backbone': False},
            'fusion': {'type': 'cross_attention', 'embed_dim': 768},
            'decoder': {'vocab_size': 30522, 'hidden_dim': 768, 'num_heads': 8, 'num_layers': 4}
        }
    }
    
    # Initialize Model
    model = GeoNarrator(mock_config).to(device)
    
    # Setup Loss and Optimizer (Ignoring the <PAD> token index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    print("Training pipeline ready. Waiting for dataloader...")
    # NOTE: To actually run this, you will instantiate your GeoNarratorDataset 
    # and pass it to a DataLoader here.