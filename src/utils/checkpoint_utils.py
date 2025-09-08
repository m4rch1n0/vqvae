"""
Utilities for loading VAE models from checkpoints with auto-detected architectures
Handles different checkpoint formats and VAE configurations automatically
"""
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from src.models.vae import VAE


def auto_detect_vae_config(state_dict: Dict) -> Dict:
    """Auto-detect VAE architecture from checkpoint state_dict
    
    Args:
        state_dict: Model state dictionary from checkpoint
        
    Returns:
        Dictionary with VAE configuration parameters
    """
    
    # Detect input channels from first conv layer
    in_channels = state_dict.get("encoder.conv_layers.0.weight", torch.zeros(1, 1, 1, 1)).shape[1]
    
    # Detect encoder channels from conv layers  
    enc_channels = []
    i = 0
    while f"encoder.conv_layers.{i*3}.weight" in state_dict:
        enc_channels.append(state_dict[f"encoder.conv_layers.{i*3}.weight"].shape[0])
        i += 1
    
    # Use defaults if detection fails
    if not enc_channels:
        enc_channels = [32, 64, 128]
    
    return {
        'in_channels': in_channels,
        'enc_channels': tuple(enc_channels),
        'dec_channels': tuple(reversed(enc_channels)),
        'norm_type': "batch" if "encoder.conv_layers.1.running_mean" in state_dict else "none",
        'output_image_size': 32 if in_channels == 3 else 28  # Simple rule: 3ch=CIFAR=32x32, 1ch=MNIST/Fashion=28x28
    }


def extract_state_dict(checkpoint: Dict) -> Dict:
    """Extract state_dict from different checkpoint formats
    
    Args:
        checkpoint: Raw checkpoint dictionary
        
    Returns:
        Clean state_dict ready for model loading
    """
    return checkpoint.get('model_state_dict') or checkpoint.get('model') or checkpoint


def load_vae_from_checkpoint(
    checkpoint_path: str, 
    latent_dim: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = True
) -> Tuple[Optional[VAE], Dict]:
    """Load VAE model from checkpoint with auto-detected architecture
    
    Args:
        checkpoint_path: Path to checkpoint file
        latent_dim: Latent dimension (auto-detected if None) 
        device: Device to load model on
        verbose: Print loading details
        
    Returns:
        (model, config) tuple, or (None, {}) if loading fails
    """
    if not Path(checkpoint_path).exists():
        if verbose:
            print(f"Checkpoint not found: {checkpoint_path}")
        return None, {}
    
    try:
        # Handle PyTorch 2.6+ weights_only default change
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        state_dict = extract_state_dict(checkpoint)
        
        # Auto-detect architecture from checkpoint
        config = auto_detect_vae_config(state_dict)
        
        # Auto-detect latent_dim if not provided
        if latent_dim is None:
            # Try to get from encoder fc layers
            mu_key = "encoder.fc_mu.weight"
            if mu_key in state_dict:
                latent_dim = state_dict[mu_key].shape[0]
            else:
                latent_dim = 128  # Default fallback
        
        config['latent_dim'] = latent_dim
        
        if verbose:
            print(f"Auto-detected: {config['in_channels']}ch, {config['enc_channels']}, "
                  f"{config['output_image_size']}x{config['output_image_size']}, {config['norm_type']}, "
                  f"latent_dim={latent_dim}")
        
        # Create and load VAE
        vae = VAE(**config).to(device).eval()
        vae.load_state_dict(state_dict)
        
        if verbose:
            print(f"VAE loaded successfully from: {checkpoint_path}")
        
        return vae, config
        
    except Exception as e:
        if verbose:
            print(f"Error loading VAE: {e}")
        return None, {}


def get_vae_decoder(checkpoint_path: str, latent_dim: Optional[int] = None, device: str = "cpu") -> Optional[torch.nn.Module]:
    """Quick helper to get just the decoder from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        latent_dim: Latent dimension (auto-detected if None)
        device: Device to load on
        
    Returns:
        Decoder module or None if loading fails
    """
    vae, _ = load_vae_from_checkpoint(checkpoint_path, latent_dim, device, verbose=False)
    return vae.decoder if vae is not None else None


# Legacy compatibility functions for existing scripts
def load_decoder(checkpoint_path: str, latent_dim: int, device: str = "cpu"):
    """Legacy compatibility function for experiments/geo scripts"""
    return get_vae_decoder(checkpoint_path, latent_dim, device)


def auto_detect_vae_config_legacy(state_dict):
    """Legacy compatibility function for experiments/geo scripts"""  
    return auto_detect_vae_config(state_dict)
