"""
Utility modules for the VQ-VAE project
"""

from .checkpoint_utils import (
    auto_detect_vae_config,
    extract_state_dict,
    load_vae_from_checkpoint,
    get_vae_decoder,
    load_decoder,  # Legacy compatibility
    auto_detect_vae_config_legacy,  # Legacy compatibility
)

__all__ = [
    'auto_detect_vae_config',
    'extract_state_dict', 
    'load_vae_from_checkpoint',
    'get_vae_decoder',
    'load_decoder',
    'auto_detect_vae_config_legacy',
]