from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels: int = 1, channels=(32, 64, 128), latent_dim: int = 16, norm_type: str = "none"):
        super().__init__()
        layers = []
        prev = input_channels
        norm_type = (norm_type or "none").lower()
        for ch in channels:
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1))
            if norm_type == "batch":
                layers.append(nn.BatchNorm2d(ch))
            elif norm_type == "group":
                # Use a reasonable default for groups while ensuring divisibility
                num_groups = max(1, min(32, ch))
                while ch % num_groups != 0 and num_groups > 1:
                    num_groups -= 1
                layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=ch))
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
    def __init__(self, out_channels: int = 1, channels=(128, 64, 32), latent_dim: int = 16, output_image_size: int = 28, norm_type: str = "none"):
        super().__init__()
        self.fc = nn.Linear(latent_dim, channels[0] * 4 * 4)
        norm_type = (norm_type or "none").lower()
        # First upsampling step: 4 -> 7 (MNIST 28) or 8 (CIFAR10 32)
        out_pad = 1 if int(output_image_size) == 32 else 0
        deconv1_layers = [
            nn.ConvTranspose2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, output_padding=out_pad),
        ]
        if norm_type == "batch":
            deconv1_layers.append(nn.BatchNorm2d(channels[1]))
        elif norm_type == "group":
            num_groups = max(1, min(32, channels[1]))
            while channels[1] % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            deconv1_layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=channels[1]))
        deconv1_layers.append(nn.ReLU(inplace=True))
        self.deconv1 = nn.Sequential(*deconv1_layers)

        # Next steps double spatial dims
        deconv2_layers = [
            nn.ConvTranspose2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
        ]
        if norm_type == "batch":
            deconv2_layers.append(nn.BatchNorm2d(channels[2]))
        elif norm_type == "group":
            num_groups = max(1, min(32, channels[2]))
            while channels[2] % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            deconv2_layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=channels[2]))
        deconv2_layers.append(nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(*deconv2_layers)
        self.out = nn.ConvTranspose2d(channels[2], out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), -1, 4, 4)
        h = self.deconv1(h)
        h = self.deconv2(h)
        x_logits = self.out(h)
        return x_logits


class VAE(nn.Module):
    def __init__(self, in_channels=1, enc_channels=(32,64,128), dec_channels=(128,64,32), latent_dim=16, recon_loss="bce", output_image_size: int = 28, norm_type: str = "none", mse_use_sigmoid: bool = True):
        super().__init__()
        self.encoder = Encoder(in_channels, enc_channels, latent_dim, norm_type=norm_type)
        self.decoder = Decoder(in_channels, dec_channels, latent_dim, output_image_size=output_image_size, norm_type=norm_type)
        assert recon_loss in {"bce", "mse"}  # must assure that the loss is either binary cross entropy or mean squared error
        self.recon_loss = recon_loss
        self.mse_use_sigmoid = bool(mse_use_sigmoid)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # reparameterization trick

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decoder(z)
        return x_logits, mu, logvar, z

    def loss(self, x: torch.Tensor, x_logits: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
        """Compute ELBO = recon + beta * KL.

        If recon_loss == "bce", use numerically-stable BCE-with-logits.
        If recon_loss == "mse", optionally apply sigmoid on logits and compute MSE.
        """
        if self.recon_loss == "bce":
            recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum') / x.size(0)
        else:
            if self.mse_use_sigmoid:
                recon = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum') / x.size(0)
            else:
                recon = F.mse_loss(x_logits, x, reduction='sum') / x.size(0)
        # KL between diagonal Gaussians
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        elbo = recon + beta * kl
        return elbo, recon, kl