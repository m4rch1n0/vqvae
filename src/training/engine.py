from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.vae import VAE
from src.utils.latents import save_latents


class TrainingEngine:
    def __init__(
        self,
        model: VAE,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int, num_epochs: int, beta: float) -> Tuple[float, float, float]:
        """Run a single epoch over a DataLoader and return averaged metrics."""
        self.model.train() if train else self.model.eval()
        total, total_recon, total_kl = 0.0, 0.0, 0.0
        steps = 0
        desc = "Train" if train else "Val"
        pbar = tqdm(loader, desc=f"{desc} [{epoch}/{num_epochs}]")
        for x, _ in pbar:
            x = x.to(self.device)
            x_logits, mu, logvar, _ = self.model(x)
            loss, recon, kl = self.model.loss(x, x_logits, mu, logvar, beta=beta)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            total += float(loss.item())
            total_recon += float(recon.item())
            total_kl += float(kl.item())
            steps += 1
            pbar.set_postfix({
                'loss': f"{total/steps:.4f}",
                'recon': f"{total_recon/steps:.4f}",
                'kl': f"{total_kl/steps:.4f}",
            })

        return total/steps, total_recon/steps, total_kl/steps

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stop: int,
        checkpoint_dir,
        logger,
        output_dir,
        save_latents_flag: bool,
        kl_anneal_epochs: int = 0,
        beta: float = 1.0,
    ) -> None:
        """Train for num_epochs with early stopping"""
        best_val = float('inf')
        no_improve = 0
        num_pixels = None
        
        # Create directories only if provided
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            # KL Annealing: linearly increase beta from 0 to target beta
            current_beta = beta * min(1.0, epoch / kl_anneal_epochs) if kl_anneal_epochs > 0 else beta

            print(f"Epoch {epoch}/{num_epochs} (beta={current_beta:.4f})")
            train_loss, train_recon, train_kl = self._run_epoch(train_loader, train=True, epoch=epoch, num_epochs=num_epochs, beta=current_beta)
            val_loss, val_recon, val_kl = self._run_epoch(val_loader, train=False, epoch=epoch, num_epochs=num_epochs, beta=current_beta)

            # Infer number of pixels once (C*H*W) for per-pixel metrics
            if num_pixels is None:
                x_sample, _ = next(iter(val_loader))
                num_pixels = int(x_sample[0].numel())
                del x_sample

            if logger is not None:
                metrics = {
                    'train_loss': train_loss,
                    'train_recon': train_recon,
                    'train_kl': train_kl,
                    'val_loss': val_loss,
                    'val_recon': val_recon,
                    'val_kl': val_kl,
                    'beta': current_beta,
                }
                if num_pixels and num_pixels > 0:
                    metrics.update({
                        'train_recon_per_pixel': train_recon / num_pixels,
                        'val_recon_per_pixel': val_recon / num_pixels,
                    })
                logger.log_metrics(metrics, step=epoch)

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                # Save checkpoint only if directory is provided
                if checkpoint_dir is not None:
                    torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch}, checkpoint_dir / 'best.pt')
            else:
                no_improve += 1
                if early_stop and no_improve >= early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if save_latents_flag and output_dir is not None:
            save_latents(self.model, train_loader, self.device, output_dir / 'latents_train')
            save_latents(self.model, val_loader, self.device, output_dir / 'latents_val')

        if output_dir is not None:
            self._save_recon_grid(val_loader, output_dir, logger)

        # Save latest checkpoint at the end as well, if directory is provided
        if checkpoint_dir is not None:
            torch.save({'model_state_dict': self.model.state_dict(), 'epoch': num_epochs}, checkpoint_dir / 'latest.pt')

    def _save_recon_grid(self, val_loader: DataLoader, output_dir, logger) -> None:
        """Generate and save a comparison grid of original vs reconstructed images."""
        if output_dir is None:
            return
            
        self.model.eval()
        import torchvision.utils as vutils
        import torchvision
        x, _ = next(iter(val_loader))
        x = x.to(self.device)
        with torch.no_grad():
            x_logits, _, _, _ = self.model(x)
            x_rec = torch.sigmoid(x_logits)
        grid = vutils.make_grid(torch.cat([x[:8], x_rec[:8]], dim=0), nrow=8)
        img_path = output_dir / 'recon_grid.png'
        torchvision.utils.save_image(grid, img_path)
        if logger is not None:
            logger.log_artifact(img_path)