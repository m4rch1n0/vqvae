import torch
import torch.nn as nn
import torch.nn.functional as F

#  ResNet-style blocks
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
        )
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=3, hidden=256, z_ch=128, n_res=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden//2, 4, 2, 1), nn.ReLU(True),   # 32->16
            nn.Conv2d(hidden//2, hidden, 4, 2, 1), nn.ReLU(True),  # 16->8
            nn.Conv2d(hidden, z_ch, 3, 1, 1),
        )
        self.res = nn.Sequential(*[ResBlock(z_ch) for _ in range(n_res)])
        self.out = nn.Conv2d(z_ch, z_ch, 1)
    def forward(self, x):
        h = self.stem(x)
        h = self.res(h)
        z_e = self.out(h)
        return z_e  # (B, z_ch, 8, 8) per CIFAR-10

class Decoder(nn.Module):
    def __init__(self, out_ch=3, hidden=256, z_ch=128, n_res=2):
        super().__init__()
        self.inp = nn.Conv2d(z_ch, z_ch, 1)
        self.res = nn.Sequential(*[ResBlock(z_ch) for _ in range(n_res)])
        self.head = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(z_ch, hidden, 4, 2, 1), nn.ReLU(True),  # 8->16
            nn.ConvTranspose2d(hidden, hidden//2, 4, 2, 1), nn.ReLU(True),  # 16->32
            nn.Conv2d(hidden//2, out_ch, 1),
            nn.Tanh(),  # output in [-1,1]
        )
    def forward(self, z_q):
        h = self.inp(z_q)
        h = self.res(h)
        x = self.head(h)
        return x

#  Vector Quantizer EMA
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_codes=512, code_dim=128, decay=0.99, eps=1e-5, beta=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta

        embed = torch.randn(n_codes, code_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", embed.clone())

        self.decay = decay
        self.eps = eps

    def forward(self, z_e):
        # z_e: (B, C, H, W)
        B, C, H, W = z_e.shape
        z = z_e.permute(0,2,3,1).contiguous()  # (B,H,W,C)
        flat = z.view(-1, C)                   # (BHW, C)

        # distance to the centroids
        # ||x - e||^2 = x^2 + e^2 - 2xe
        flat = flat.float()
        embed = self.embed.float()
        d = (flat**2).sum(1, keepdim=True) - 2 * flat @ embed.t() + (embed**2).sum(1)

        idx = torch.argmin(d, dim=1)          # (BHW,)
        z_q = self.embed.index_select(0, idx).view(B, H, W, C)
        z_q = z_q.permute(0,3,1,2).contiguous()  # (B,C,H,W)

        # EMA updates (no grad)
        if self.training:
            with torch.no_grad():
                one_hot = torch.zeros(idx.size(0), self.n_codes, device=z_e.device)
                one_hot.scatter_(1, idx.view(-1,1), 1)
                # cluster sizes
                cluster_size = one_hot.sum(0)
                self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1-self.decay)

                # embed_avg
                embed_sum = flat.t() @ one_hot
                self.embed_avg.mul_(self.decay).add_(embed_sum.t(), alpha=1-self.decay)

                # normalize with numerical guards
                n = self.cluster_size.sum()
                denom = n + self.n_codes * self.eps
                cluster_size = (self.cluster_size + self.eps) / denom * n
                # avoid division by zero
                cluster_size_safe = cluster_size.unsqueeze(1).clamp_min(self.eps)
                embed_normalized = self.embed_avg / cluster_size_safe
                # numerical cleanup
                embed_normalized = torch.nan_to_num(embed_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
                embed_normalized = embed_normalized.clamp_(-2.0, 2.0)
                self.embed.copy_(embed_normalized)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # losses
        loss_commit = F.mse_loss(z_q_st.detach().float(), z_e.float())
        loss = self.beta * loss_commit
        return z_q_st, loss, idx.view(B, H, W), z_q, z_e

    def reseed_dead_codes(self, min_count: int = 5, sample_bank: torch.Tensor = None):
        """
        Reseed codes whose EMA cluster_size is below min_count using vectors
        sampled from a provided latent sample bank (shape: (N, C)).
        Returns the number of codes reseeded.
        """
        with torch.no_grad():
            if sample_bank is None or sample_bank.numel() == 0:
                return 0
            dead_mask = self.cluster_size < float(min_count)
            num_dead = int(dead_mask.sum().item())
            if num_dead == 0:
                return 0

            num_bank, code_dim = sample_bank.size(0), sample_bank.size(1)
            if code_dim != self.code_dim:
                # dimension mismatch, skip safely
                return 0
            num_pick = min(num_dead, num_bank)
            perm = torch.randperm(num_bank, device=sample_bank.device)[:num_pick]
            new_vecs = sample_bank[perm]

            dead_idx = dead_mask.nonzero(as_tuple=False).view(-1)[:num_pick]
            self.embed[dead_idx] = new_vecs.to(self.embed.dtype)
            self.embed_avg[dead_idx] = self.embed[dead_idx]
            self.cluster_size[dead_idx] = float(min_count)
            return int(num_pick)

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, z_channels=128, hidden=256, n_res_blocks=2,
                 n_codes=512, beta=0.25, ema_decay=0.99, ema_eps=1e-5):
        super().__init__()
        self.enc = Encoder(in_ch=in_channels, hidden=hidden, z_ch=z_channels, n_res=n_res_blocks)
        self.quant = VectorQuantizerEMA(n_codes=n_codes, code_dim=z_channels,
                                        decay=ema_decay, eps=ema_eps, beta=beta)
        self.dec = Decoder(out_ch=in_channels, hidden=hidden, z_ch=z_channels, n_res=n_res_blocks)

    def forward(self, x):
        z_e = self.enc(x)
        z_q_st, loss_vq, idx, z_q, z_e = self.quant(z_e)
        x_rec = self.dec(z_q_st)
        return x_rec, loss_vq, idx, z_q, z_e
