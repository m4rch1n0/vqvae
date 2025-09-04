from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_norm_layer(norm_type: str, num_channels: int) -> nn.Module:
    """Create normalization layer based on type."""
    norm_type = (norm_type or "none").lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "group":
        groups = max(1, min(32, num_channels))
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    else:
        return nn.Identity()


class SpatialEncoder(nn.Module):
    """Convolutional encoder that outputs a spatial latent distribution."""

    def __init__(self, input_channels: int, channels: Tuple[int, ...],
                 latent_dim: int, norm_type: str):
        super().__init__()
        layers = []
        prev_channels = input_channels
        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_channels, ch, 3, stride=2, padding=1),
                _get_norm_layer(norm_type, ch),
                nn.ReLU(inplace=True),
            ])
            prev_channels = ch
        self.conv_layers = nn.Sequential(*layers)
        
        self.fc_mu = nn.Conv2d(channels[-1], latent_dim, 1)
        self.fc_logvar = nn.Conv2d(channels[-1], latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_layers(x)
        return self.fc_mu(h), self.fc_logvar(h)


class SpatialDecoder(nn.Module):
    """Convolutional decoder that reconstructs from a spatial latent grid."""

    def __init__(self, out_channels: int, channels: Tuple[int, ...],
                 latent_dim: int, output_image_size: int, norm_type: str):
        super().__init__()
        
        self.conv_in = nn.Conv2d(latent_dim, channels[0], 1)

        layers = []
        # Start from 4x4 spatial grid from encoder
        layers.extend([
            nn.ConvTranspose2d(channels[0], channels[1], 4, stride=2, padding=1), # -> 8x8
            _get_norm_layer(norm_type, channels[1]),
            nn.ReLU(inplace=True),
        ])
        layers.extend([
            nn.ConvTranspose2d(channels[1], channels[2], 4, stride=2, padding=1), # -> 16x16
            _get_norm_layer(norm_type, channels[2]),
            nn.ReLU(inplace=True),
        ])
        
        # Final layer to get to 28x28
        layers.append(nn.ConvTranspose2d(channels[2], out_channels, 4, stride=2, padding=3))

        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        return self.deconv_layers(h)


class SpatialVAE(nn.Module):
    """A VAE with a spatial latent representation suitable for VQ."""

    def __init__(self, in_channels, enc_channels, dec_channels, latent_dim,
                 recon_loss, output_image_size, norm_type, **kwargs):
        super().__init__()
        self.encoder = SpatialEncoder(in_channels, enc_channels, latent_dim, norm_type)
        self.decoder = SpatialDecoder(in_channels, dec_channels, latent_dim, output_image_size, norm_type)
        
        assert recon_loss in {"bce", "mse"}
        self.recon_loss = recon_loss
        self.mse_use_sigmoid = kwargs.get('mse_use_sigmoid', True)
        self._step = 0

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decoder(z)
        return x_logits, mu, logvar, z

    def loss(self, x, x_logits, mu, logvar, beta: float, **kwargs) -> Tuple[torch.Tensor, ...]:
        batch_size = x.size(0)
        
        # Reconstruction loss
        if self.recon_loss == "bce":
            recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum') / batch_size
        else:
            x_pred = torch.sigmoid(x_logits) if self.mse_use_sigmoid else x_logits
            recon = F.mse_loss(x_pred, x, reduction='sum') / batch_size

        # KL divergence (spatial)
        kl_per_pixel = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl_per_pixel.sum(dim=[1, 2, 3]).mean()

        total_loss = recon + beta * kl
        return total_loss, recon, kl
