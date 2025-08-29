# Transformer Implementation

## Overview

GPT-style transformer for autoregressive generation of discrete codes from post-hoc VQ-VAE codebooks. Enables image generation without joint training.

## Core Functions

### CausalSelfAttention
**Purpose**: Multi-head attention with causal masking for autoregressive generation
**Key Features**:
- Prevents seeing future tokens during training
- Uses lower triangular mask for position-aware attention
- Caches causal mask to avoid recomputation
- Supports configurable dropout for attention and residual connections

### TransformerBlock
**Purpose**: Standard transformer block with pre-norm architecture
**Structure**: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
**Benefits**: Pre-norm improves training stability, residual connections help gradient flow

### GPT
**Purpose**: Main transformer model for sequence generation
**Components**: Token embeddings, positional encodings, class embeddings (optional), transformer blocks
**Initialization**: GPT-2 style weight initialization (normal distribution, std=0.02)

### generate()
**Purpose**: Autoregressive token generation
**Parameters**: model, start tokens, max_new_tokens, temperature, top_k, top_p, classes
**Process**: Iteratively samples next tokens using specified sampling strategy

## Configuration

**GPTConfig**: Centralized configuration for all model parameters
- vocab_size: Number of discrete tokens (codebook size + special tokens)
- seq_len: Maximum sequence length (H×W for image codes)
- embed_dim: Hidden dimension for embeddings and attention
- num_layers: Number of transformer blocks
- class_conditional: Whether to support class-conditional generation

## Usage Example

```python
from src.models.transformer import GPT, GPTConfig

config = GPTConfig(
    vocab_size=1000,
    seq_len=1024,  # 32×32 for CIFAR-10
    embed_dim=256,
    num_layers=8,
    class_conditional=True,
    num_classes=10
)

model = GPT(config)
logits, loss = model(input_ids, labels=labels, classes=classes)
```

## Key Implementation Details

**Causal Masking**: Efficient triangular mask caching for variable sequence lengths
**Position Encoding**: Learned positional embeddings for sequence positions
**Class Conditioning**: Optional class embeddings added to token representations
**Memory Optimization**: No automatic mixed precision (ROCm compatibility)
**Gradient Control**: Configurable gradient clipping for training stability
