import torch
import torch.nn as nn

# Import the modules we just built
from models.encoders import SiameseEncoder
from models.fusion import FeatureFusion
from models.vlm_decoder import VLMDecoder

class GeoNarrator(nn.Module):
    """
    The main wrapper class that connects the Vision Encoder, Difference Fusion, 
    and Language Decoder into a single end-to-end Vision Language Model.
    """
    def __init__(self, config):
        """
        Initializes the model dynamically using your default.yaml configuration.
        """
        super().__init__()
        
        # 1. Siamese Vision Backbone
        self.encoder = SiameseEncoder(
            model_name=config['model']['encoder']['type'],
            pretrained=config['model']['encoder']['pretrained'],
            freeze_backbone=config['model']['encoder']['freeze_backbone']
        )
        
        # 2. Difference Fusion Module
        self.fusion = FeatureFusion(
            fusion_type=config['model']['fusion']['type'],
            embed_dim=config['model']['fusion']['embed_dim']
        )
        
        # 3. Autoregressive Language Decoder
        self.decoder = VLMDecoder(
            vocab_size=config['model']['decoder']['vocab_size'],
            embed_dim=config['model']['decoder']['hidden_dim'],
            num_heads=config['model']['decoder']['num_heads'],
            num_layers=config['model']['decoder']['num_layers']
        )
        
    def forward(self, image_A, image_B, text_tokens, padding_mask=None):
        """
        TRAINING PASS: Processes the entire text sequence at once using "Teacher Forcing".
        The causal mask in the decoder ensures the model cannot look ahead at future words.
        """
        # Step 1: Extract visual features
        feat_A, feat_B = self.encoder(image_A, image_B)
        
        # Step 2: Compute the spatial difference
        change_memory = self.fusion(feat_A, feat_B)
        
        # Step 3: Predict the sentence probabilities
        logits = self.decoder(text_tokens, change_memory, padding_mask)
        
        return logits
        
    @torch.no_grad()
    def generate(self, image_A, image_B, start_token_id, eos_token_id, max_length=50):
        """
        INFERENCE PASS: Autoregressively generates a sentence word-by-word 
        when testing the model on new, unseen satellite images.
        """
        self.eval() # Turn off dropout/batchnorm for inference
        device = image_A.device
        
        # 1. Extract and fuse visual features exactly once
        feat_A, feat_B = self.encoder(image_A, image_B)
        change_memory = self.fusion(feat_A, feat_B)
        
        # 2. Start the sentence with the <SOS> (Start of Sequence) token
        generated_tokens = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
        
        # 3. The Autoregressive Loop
        for _ in range(max_length):
            # Pass the currently generated sequence to the decoder
            logits = self.decoder(generated_tokens, change_memory)
            
            # Get the highest probability for the very last word (Greedy Decoding)
            next_word_logits = logits[:, -1, :] 
            next_word_id = torch.argmax(next_word_logits, dim=-1).unsqueeze(1)
            
            # Append the newly predicted word to our running sequence
            generated_tokens = torch.cat((generated_tokens, next_word_id), dim=1)
            
            # Stop generating if the model outputs the <EOS> (End of Sequence) token
            if next_word_id.item() == eos_token_id:
                break
                
        return generated_tokens