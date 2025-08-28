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
        # Use largest number of groups (≤32) that divides num_channels
        groups = max(1, min(32, num_channels))
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    else:
        return nn.Identity()


class Encoder(nn.Module):
    """Convolutional encoder that outputs latent mean and log-variance."""

    def __init__(self, input_channels: int = 1, channels=(32, 64, 128),
                 latent_dim: int = 16, norm_type: str = "none"):
        super().__init__()

        # Build convolutional layers
        layers = []
        prev_channels = input_channels
        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_channels, ch, 3, stride=2, padding=1),
                _get_norm_layer(norm_type, ch),
                nn.ReLU(inplace=True)
            ])
            prev_channels = ch

        self.conv_layers = nn.Sequential(*layers)

        # Calculate flattened feature dimension
        self.feature_dim = channels[-1] * 4 * 4
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Convolutional decoder that reconstructs images from latent space."""

    def __init__(self, out_channels: int = 1, channels=(128, 64, 32),
                 latent_dim: int = 16, output_image_size: int = 28, norm_type: str = "none"):
        super().__init__()

        self.fc = nn.Linear(latent_dim, channels[0] * 4 * 4)

        # First deconvolution: 4x4 -> 8x8 (CIFAR) or 7x7 (MNIST)
        output_padding = 1 if output_image_size == 32 else 0
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[1], 3, stride=2, padding=1, output_padding=output_padding),
            _get_norm_layer(norm_type, channels[1]),
            nn.ReLU(inplace=True)
        )

        # Second deconvolution: upsample to target size
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[2], 4, stride=2, padding=1),
            _get_norm_layer(norm_type, channels[2]),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.output_layer = nn.ConvTranspose2d(channels[2], out_channels, 4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), -1, 4, 4)
        h = self.deconv1(h)
        h = self.deconv2(h)
        return self.output_layer(h)


class VAE(nn.Module):
    """Convolutional Variational Autoencoder with configurable regularization."""

    def __init__(self, in_channels=1, enc_channels=(32, 64, 128), dec_channels=(128, 64, 32),
                 latent_dim=16, recon_loss="bce", output_image_size: int = 28,
                 norm_type: str = "none", mse_use_sigmoid: bool = True,
                 free_bits_default: float = 0.5, capacity_max_default: float = 15.0,
                 capacity_anneal_steps_default: int = 50_000, capacity_mode_default: str = "abs"):
        super().__init__()

        # Core architecture
        self.encoder = Encoder(in_channels, enc_channels, latent_dim, norm_type)
        self.decoder = Decoder(in_channels, dec_channels, latent_dim, output_image_size, norm_type)

        # Reconstruction loss configuration
        assert recon_loss in {"bce", "mse"}, f"recon_loss must be 'bce' or 'mse', got {recon_loss}"
        self.recon_loss = recon_loss
        self.mse_use_sigmoid = mse_use_sigmoid

        # Default regularization parameters
        self.free_bits_default = free_bits_default
        self.capacity_max_default = capacity_max_default
        self.capacity_anneal_steps_default = capacity_anneal_steps_default
        self.capacity_mode_default = capacity_mode_default

        # Training step counter
        self._step = 0

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decoder(z)
        return x_logits, mu, logvar, z

    def _compute_reconstruction_loss(self, x_logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss based on configured type."""
        batch_size = x.size(0)

        if self.recon_loss == "bce":
            return F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum') / batch_size
        else:  # MSE
            x_pred = torch.sigmoid(x_logits) if self.mse_use_sigmoid else x_logits
            return F.mse_loss(x_pred, x, reduction='sum') / batch_size

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor,
                        free_bits: float | None = None) -> torch.Tensor:
        """Compute KL divergence loss with optional free-bits regularization."""
        # KL per dimension: -0.5 * (1 + logvar - mu² - exp(logvar))
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        # Apply free-bits constraint if specified
        if free_bits is not None:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

        return kl_per_dim.sum(dim=1).mean()

    def _compute_capacity_target(self, capacity_max: float, capacity_anneal_steps: int,
                                step: int) -> float:
        """Compute current capacity target based on training step."""
        progress = min(1.0, step / max(1, capacity_anneal_steps))
        return capacity_max * progress

    def loss(self, x, x_logits, mu, logvar, *, beta: float = 1.0,
             free_bits: float | None = None, capacity_max: float | None = None,
             capacity_anneal_steps: int | None = None, step: int | None = None,
             capacity_mode: str | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO loss with optional regularization techniques.

        Args:
            free_bits: Minimum KL per dimension to prevent posterior collapse
            capacity_max: Target KL capacity for annealing
            capacity_anneal_steps: Steps over which to anneal capacity
            capacity_mode: "abs" for absolute difference, "clipped" for one-sided
        """
        # Use defaults if parameters not provided
        free_bits = self.free_bits_default if free_bits is None else free_bits
        capacity_max = self.capacity_max_default if capacity_max is None else capacity_max
        capacity_anneal_steps = self.capacity_anneal_steps_default if capacity_anneal_steps is None else capacity_anneal_steps
        capacity_mode = self.capacity_mode_default if capacity_mode is None else capacity_mode

        # Auto-increment step counter if not provided
        if step is None:
            step = self._step
            self._step += 1

        # Compute losses
        recon_loss = self._compute_reconstruction_loss(x_logits, x)
        kl_loss = self._compute_kl_loss(mu, logvar, free_bits)

        # Apply capacity annealing if configured
        if capacity_max > 0 and capacity_anneal_steps > 0:
            capacity_target = self._compute_capacity_target(capacity_max, capacity_anneal_steps, step)

            if capacity_mode == "abs":
                kl_regulated = torch.abs(kl_loss - capacity_target)
            else:  # "clipped"
                kl_regulated = torch.clamp(kl_loss - capacity_target, min=0.0)

            total_loss = recon_loss + beta * kl_regulated
        else:
            total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss