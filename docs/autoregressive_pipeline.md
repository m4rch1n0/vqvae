# Autoregressive Training Pipeline

## Overview

Complete pipeline for training autoregressive models on discrete codes and generating images.

## Core Functions

### CodeSequenceDataset
**Purpose**: Converts discrete codes to training sequences
**Input**: .npy files with integer codes (N, H×W)
**Output**: PyTorch Dataset with input_ids and labels
**Features**: Automatic BOS/EOS/PAD token assignment, class label support, invalid code mapping

### pad_collate
**Purpose**: Collate function for batching variable-length sequences
**Process**: Pads sequences to maximum length in batch
**Special handling**: Pads input_ids with pad_id, labels with -100 (loss ignore)

### sample_next_token
**Purpose**: Implements various decoding strategies
**Methods**: Temperature scaling, top-k sampling, top-p (nucleus) sampling
**Output**: Single token sampled from probability distribution

### generate_autoregressive
**Purpose**: Generates complete token sequences
**Process**: Iteratively samples next tokens using specified strategy
**Memory management**: Uses only last seq_len tokens to avoid overflow

### decode_codes_to_images
**Purpose**: Converts discrete codes back to images
**Process**: Maps codes to latent codes via codebook, decodes with VAE
**Output**: Generated image tensors

## Training Pipeline

### train_transformer.py
**Purpose**: Main training script for transformer models
**Features**: Hydra configuration, MLflow logging, automatic token assignment
**Output**: Model checkpoints, training logs, experiment tracking

### Configuration Files
**Location**: configs/transformer/
**Datasets**: cifar10_small.yaml, fashion_small.yaml, mnist_small.yaml
**Parameters**: seq_len, embed_dim, num_layers, batch_size, learning rate

## Usage

### Training
```bash
python src/training/train_transformer.py --config-name transformer/cifar10_small
```

### Generation
```bash
python src/generation/generate_images.py \
  --transformer_ckpt experiments/transformer/cifar10_small/checkpoints/best.pt \
  --vae_ckpt experiments/vae_cifar10/checkpoints/best.pt \
  --codebook_pt experiments/geo/test_optimized_cifar10/codebook.pt
```

## Data Flow

1. Load discrete codes from codes.npy
2. Convert to training sequences with BOS/EOS/PAD
3. Train transformer on next-token prediction
4. Generate autoregressive sequences
5. Map codes to latents via codebook
6. Decode to images using VAE

## Key Features

**Automatic Token Management**: BOS/EOS/PAD IDs assigned after base vocabulary
**Class Conditioning**: Support for class-conditional generation
**Integration**: Seamless connection with existing codebook infrastructure
