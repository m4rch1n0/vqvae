from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_group_count(num_channels: int) -> int:
    """Largest number of groups (<=32) that divides num_channels."""
    groups = max(1, min(32, num_channels))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def make_norm(norm_type: str, num_channels: int) -> nn.Module:
    """Return a normalization layer matching norm_type, or Identity if none."""
    key = (norm_type or "none").lower()
    if key == "batch":
        return nn.BatchNorm2d(num_channels)
    if key == "group":
        return nn.GroupNorm(num_groups=compute_group_count(num_channels), num_channels=num_channels)
    return nn.Identity()


class Encoder(nn.Module):
    """Simple stride-2 conv encoder producing mean and log-variance."""

    def __init__(self, input_channels: int = 1, channels=(32, 64, 128), latent_dim: int = 16, norm_type: str = "none"):
        super().__init__()
        layers = []
        prev = input_channels
        for ch in channels:
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1))
            layers.append(make_norm(norm_type, ch))
            layers.append(nn.ReLU(inplace=True))
            prev = ch
        self.conv = nn.Sequential(*layers)

        feat_dim = channels[-1] * 4 * 4
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Two deconv blocks followed by output layer to match image size."""

    def __init__(self, out_channels: int = 1, channels=(128, 64, 32), latent_dim: int = 16, output_image_size: int = 28, norm_type: str = "none"):
        super().__init__()
        self.fc = nn.Linear(latent_dim, channels[0] * 4 * 4)
        # First upsampling step: 4 -> 7 (MNIST 28) or 8 (CIFAR10 32)
        out_pad = 1 if int(output_image_size) == 32 else 0
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, output_padding=out_pad),
            make_norm(norm_type, channels[1]),
            nn.ReLU(inplace=True),
        )

        # Next steps double spatial dims
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
            make_norm(norm_type, channels[2]),
            nn.ReLU(inplace=True),
        )
        self.out = nn.ConvTranspose2d(channels[2], out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), -1, 4, 4)
        h = self.deconv1(h)
        h = self.deconv2(h)
        x_logits = self.out(h)
        return x_logits


class VAE(nn.Module):
    """Convolutional VAE with configurable reconstruction loss and KL regularization.

    Free-bits and capacity annealing are controlled by defaults set at init, but
    can be overridden per-call via the loss(...) keyword arguments.
    """
    def __init__(self, in_channels=1, enc_channels=(32,64,128), dec_channels=(128,64,32),
                 latent_dim=16, recon_loss="bce", output_image_size: int = 28,
                 norm_type: str = "none", mse_use_sigmoid: bool = True,
                 free_bits_default: float = 0.5,
                 capacity_max_default: float = 15.0,
                 capacity_anneal_steps_default: int = 50_000,
                 capacity_mode_default: str = "abs"):
        super().__init__()
        self.encoder = Encoder(in_channels, enc_channels, latent_dim, norm_type=norm_type)
        self.decoder = Decoder(in_channels, dec_channels, latent_dim,
                               output_image_size=output_image_size, norm_type=norm_type)
        assert recon_loss in {"bce", "mse"}
        self.recon_loss = recon_loss
        self.mse_use_sigmoid = bool(mse_use_sigmoid)

        # Regularization defaults (configurable)
        self.free_bits_default = float(free_bits_default)
        self.capacity_max_default = float(capacity_max_default)
        self.capacity_anneal_steps_default = int(capacity_anneal_steps_default)
        self.capacity_mode_default = str(capacity_mode_default)
        # Internal step counter (used if caller does not pass step)
        self._step = 0

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decoder(z)
        return x_logits, mu, logvar, z

    def loss(self, x, x_logits, mu, logvar, *,
             beta: float = 1.0,
             free_bits: float | None = None,
             capacity_max: float | None = None,
             capacity_anneal_steps: int | None = None,
             step: int | None = None,
             capacity_mode: str | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO components (loss, recon, kl) with free-bits and capacity scheduling.

        - free_bits: minimum per-dimension KL (nats) to prevent collapse
        - capacity: linearly increases target KL from 0 to capacity_max over capacity_anneal_steps
        """
        # Resolve defaults
        free_bits = self.free_bits_default if free_bits is None else float(free_bits)
        capacity_max = self.capacity_max_default if capacity_max is None else float(capacity_max)
        capacity_anneal_steps = self.capacity_anneal_steps_default if capacity_anneal_steps is None else int(capacity_anneal_steps)
        capacity_mode = self.capacity_mode_default if capacity_mode is None else str(capacity_mode)
        if step is None:
            step = self._step
            self._step += 1

        # Reconstruction
        if self.recon_loss == "bce":
            recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum') / x.size(0)
        else:
            x_for_mse = torch.sigmoid(x_logits) if self.mse_use_sigmoid else x_logits
            recon = F.mse_loss(x_for_mse, x, reduction='sum') / x.size(0)

        # KL per-dimension (with free-bits clamp)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if free_bits is not None:
            kl_per_dim = torch.clamp(kl_per_dim, min=float(free_bits))
        kl = kl_per_dim.sum(dim=1).mean()

        # Capacity scheduling (0 -> capacity_max)
        C: Optional[float] = None
        if capacity_max is not None and capacity_anneal_steps is not None:
            frac = min(1.0, float(step) / float(max(1, capacity_anneal_steps)))
            C = float(capacity_max) * frac

        if C is not None:
            if capacity_mode == "abs":
                loss = recon + beta * torch.abs(kl - C)
            else:  # "clipped"
                loss = recon + beta * torch.clamp(kl - C, min=0.0)
        else:
            loss = recon + beta * kl

        return loss, recon, kl