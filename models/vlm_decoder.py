import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    Without this, the Transformer would treat the sentence as a "bag of words" rather than an ordered sequence.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Sequence Length, Embedding Dim)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

class VLMDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder that generates change narratives.
    """
    def __init__(self, vocab_size=30522, embed_dim=768, num_heads=8, num_layers=4, max_seq_len=50):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Text Embedding & Positional Encoding
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        
        # 2. Transformer Decoder Layers
        # batch_first=True makes tensor shapes (Batch, Seq, Feature) which is much easier to read
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True,
            norm_first=True # Best practice for deep transformers
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 3. Final Output Layer to predict the next word in the vocabulary
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """
        Prevents the model from "cheating" by looking ahead at future words during training.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt_tokens, fusion_memory, tgt_padding_mask=None):
        """
        Args:
            tgt_tokens: Tensor (B, Seq_Len) - The text tokens we are predicting
            fusion_memory: Tensor (B, C, H, W) - The fused change map from Step 4
            tgt_padding_mask: Tensor (B, Seq_Len) - Masks out <PAD> tokens so they aren't attended to
        """
        # Ensure memory is properly shaped for sequence processing
        # Change from spatial grid (B, C, H, W) to sequence (B, H*W, C)
        if fusion_memory.dim() == 4:
            B, C, H, W = fusion_memory.shape
            fusion_memory = fusion_memory.flatten(2).transpose(1, 2)

        # 1. Embed the target words and add positional information
        tgt = self.word_embedding(tgt_tokens) * math.sqrt(self.embed_dim)
        tgt = self.pos_encoder(tgt)
        
        # 2. Create the causal mask to hide future words
        seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)
        
        # 3. Pass through the decoder
        # 'tgt' goes into Self-Attention
        # 'fusion_memory' goes into Cross-Attention as Keys and Values
        output = self.transformer_decoder(
            tgt=tgt,
            memory=fusion_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # 4. Predict the probabilities for the next word
        logits = self.fc_out(output)
        return logits

# --- Quick Test Block ---
if __name__ == "__main__":
    print("Testing VLM Decoder...")
    decoder = VLMDecoder(vocab_size=1000, embed_dim=768)
    
    # Dummy fusion map (from Step 4) and dummy target text tokens
    dummy_fusion = torch.randn(2, 768, 8, 8) 
    dummy_text = torch.randint(0, 1000, (2, 20)) # Batch size 2, 20 words long
    
    logits = decoder(dummy_text, dummy_fusion)
    print(f"Logits Shape: {logits.shape}") # Should be [2, 20, 1000]