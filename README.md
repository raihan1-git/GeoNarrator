# GeoNarrator 
**A Dual-Encoder Vision Language Model for Satellite Image Change Narration**

![Status: WIP](https://img.shields.io/badge/Status-Work_in_Progress-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

GeoNarrator is a multimodal Deep Learning project designed to move beyond simple binary change detection in remote sensing. Instead of merely outputting a pixel mask of changed areas, GeoNarrator autoregressively generates natural language narratives explaining the *semantic meaning* of the landscape changes between temporal image pairs (e.g., *"A new residential complex was constructed in the previously empty field."*).

> **🚧 Development Status:** This project is currently in active development. The core architecture (Siamese Encoders, Cross-Attention Fusion, VLM Decoder) is fully implemented. The model is currently undergoing training and evaluation on the LEVIR-CC dataset.

##  Architecture Overview

GeoNarrator utilizes a state-of-the-art "Encoder-Neck-Decoder" pipeline built entirely from scratch in PyTorch:

1. **Siamese Vision Backbone (`timm`):** Uses pre-trained Swin Transformers (or ViTs) with shared weights to extract hierarchical spatial features from the $T_1$ (Before) and $T_2$ (After) satellite images.
2. **Difference Fusion Module:** Employs **Cross-Attention** mechanisms, allowing the $T_2$ features to query the $T_1$ features to mathematically isolate the spatial differences while retaining geographic context.
3. **Autoregressive Language Decoder:** A custom Transformer Decoder that attends to the fused "Change Embedding" to predict the descriptive sequence word-by-word, utilizing causal masking for training.

## 📁 Repository Structure

```text
GeoNarrator/
├── configs/
│   └── default.yaml       # Hyperparameters (batch size, Swin dims, LR)
├── data/
│   ├── dataset.py         # PyTorch Dataset for LEVIR-CC image pairs & captions
│   └── transforms.py      # ImageNet normalization and data augmentation
├── models/
│   ├── encoders.py        # Siamese Swin Transformer implementation
│   ├── fusion.py          # Cross-Attention and Concatenation fusion layers
│   ├── vlm_decoder.py     # Transformer Text Decoder
│   └── generator.py       # Main wrapper connecting Vision, Fusion, and Text
├── scripts/
│   ├── train.py           # Training loop with Cross-Entropy Loss optimization
│   └── inference.py       # Real-world inference on raw image pairs
├── utils/
│   └── metrics.py         # BLEU-4 and validation scoring
└── requirements.txt
