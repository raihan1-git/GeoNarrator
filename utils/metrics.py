import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download necessary NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextMetrics:
    """
    Computes evaluation metrics for the generated change narratives.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Smoothing function helps prevent BLEU scores from dropping to 0 
        # if there are no higher-order n-gram matches.
        self.chencherry = SmoothingFunction()

    def decode_tokens(self, token_ids):
        """Converts raw tensor IDs back into human-readable strings."""
        # Remove padding and special tokens
        tokens = [t for t in token_ids if t not in [self.tokenizer.pad_token_id, 
                                                    self.tokenizer.cls_token_id, 
                                                    self.tokenizer.sep_token_id]]
        return self.tokenizer.decode(tokens)

    def calculate_bleu(self, reference_text, candidate_text):
        """
        Calculates the BLEU-4 score for a single generated sentence.
        """
        # NLTK expects a list of lists for references, and a list for candidate
        ref_tokens = [nltk.word_tokenize(reference_text.lower())]
        cand_tokens = nltk.word_tokenize(candidate_text.lower())
        
        # Calculate BLEU with weights (0.25 for 1-gram, 2-gram, 3-gram, 4-gram)
        score = sentence_bleu(ref_tokens, cand_tokens, 
                              weights=(0.25, 0.25, 0.25, 0.25), 
                              smoothing_function=self.chencherry.method1)
        return score

# --- Quick Test Block ---
if __name__ == "__main__":
    metrics = TextMetrics(tokenizer=None) # Mock test without actual tokenizer
    
    reference = "A new residential building was constructed in the empty field."
    candidate_good = "A new residential complex was built in the empty field."
    candidate_bad = "The forest was removed and replaced by water."
    
    print(f"Good Match BLEU: {metrics.calculate_bleu(reference, candidate_good):.4f}")
    print(f"Bad Match BLEU:  {metrics.calculate_bleu(reference, candidate_bad):.4f}")