import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class GeoNarratorDataset(Dataset):
    """
    PyTorch Dataset class for loading LEVIR-CC image pairs and tokenizing captions.
    """
    def __init__(self, data_dir, split='train', tokenizer=None, transform=None, max_seq_length=50):
        """
        Args:
            data_dir (str): Root directory of the LEVIR-CC dataset.
            split (str): 'train', 'val', or 'test'.
            tokenizer: HuggingFace tokenizer (e.g., BertTokenizer).
            transform: torchvision transforms for image augmentations.
            max_seq_length (int): Maximum length for the tokenized text.
        """
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_length = max_seq_length
        
        # LEVIR-CC standard folder structure
        self.img_dir_A = os.path.join(data_dir, 'images', split, 'A')
        self.img_dir_B = os.path.join(data_dir, 'images', split, 'B')
        
        # Load captions JSON
        json_path = os.path.join(data_dir, 'LevirCCcaptions.json')
        with open(json_path, 'r') as f:
            all_data = json.load(f)
            
        # Filter data for the specific split
        self.data = [item for item in all_data['images'] if item['split'] == split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        filename = item['filename']
        
        # 1. Load the Before (A) and After (B) images
        path_A = os.path.join(self.img_dir_A, filename)
        path_B = os.path.join(self.img_dir_B, filename)
        
        image_A = Image.open(path_A).convert('RGB')
        image_B = Image.open(path_B).convert('RGB')
        
        # 2. Apply spatial transformations (resizing, normalization)
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            
        # 3. Process the text caption
        # LEVIR-CC provides up to 5 captions per image pair. We randomly sample one during training 
        # to increase linguistic variation, or default to the first one.
        raw_caption = item['sentences'][0]['raw'] 
        
        if self.tokenizer:
            # Tokenize and pad/truncate to max_seq_length
            encoded = self.tokenizer(
                raw_caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        else:
            input_ids = raw_caption
            attention_mask = torch.tensor([])

        return {
            'image_A': image_A,
            'image_B': image_B,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'filename': filename
        }