from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels: int = 1, channels=(32, 64, 128), latent_dim: int = 16):
        super().__init__()
        layers = []
        prev = input_channels
        for ch in channels:
            layers.extend([
                nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
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
    def __init__(self, out_channels: int = 1, channels=(128, 64, 32), latent_dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(latent_dim, channels[0] * 4 * 4)
        # 4 -> 7
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
        )
        # 7 -> 14
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # 14 -> 28
        self.out = nn.ConvTranspose2d(channels[2], out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), -1, 4, 4)
        h = self.deconv1(h)
        h = self.deconv2(h)
        x_logits = self.out(h)
        return x_logits


class VAE(nn.Module):
    def __init__(self, in_channels=1, enc_channels=(32,64,128), dec_channels=(128,64,32), latent_dim=16, recon_loss="bce"):
        super().__init__()
        self.encoder = Encoder(in_channels, enc_channels, latent_dim)
        self.decoder = Decoder(in_channels, dec_channels, latent_dim)
        assert recon_loss in {"bce", "mse"}
        self.recon_loss = recon_loss

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

    def loss(self, x: torch.Tensor, x_logits: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        """Compute ELBO = recon + KL.

        If recon_loss == "bce", use numerically-stable BCE-with-logits.
        If recon_loss == "mse", apply sigmoid on logits and compute MSE.
        """
        if self.recon_loss == "bce":
            recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum') / x.size(0)
        else:
            recon = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum') / x.size(0)
        # KL between diagonal Gaussians
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        elbo = recon + kl
        return elbo, recon, kl