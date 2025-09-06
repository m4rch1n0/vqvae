from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.spatial_vae import SpatialVAE
from src.utils.spatial_latents import save_spatial_latents
from src.eval.metrics import psnr as psnr_metric, ssim_simple


class SpatialTrainingEngine:
    def __init__(
        self,
        model: SpatialVAE,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def run_epoch(self, loader: DataLoader, train: bool, epoch: int, num_epochs: int, beta: float, grad_clip_max_norm: float = 0.0, global_step_start: int = 0) -> Tuple[float, float, float, int, float, float]:
        self.model.train() if train else self.model.eval()
        total, total_recon, total_kl = 0.0, 0.0, 0.0
        steps = 0
        desc = "Train" if train else "Val"
        pbar = tqdm(loader, desc=f"{desc} [{epoch}/{num_epochs}]")
        global_step = int(global_step_start)
        
        apply_sigmoid = (getattr(self.model, 'recon_loss', 'mse') == 'bce') or getattr(self.model, 'mse_use_sigmoid', True)
        val_psnr_sum, val_ssim_sum, val_count = 0.0, 0.0, 0
        
        for x, _ in pbar:
            x = x.to(self.device)
            if train:
                x_logits, mu, logvar, _ = self.model(x)
                loss, recon, kl = self.model.loss(x, x_logits, mu, logvar, beta=beta, step=global_step)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_max_norm)
                self.optimizer.step()
                global_step += 1
            else:
                with torch.no_grad():
                    x_logits, mu, logvar, _ = self.model(x)
                    loss, recon, kl = self.model.loss(x, x_logits, mu, logvar, beta=beta, step=global_step)
                    
                    x_rec = torch.sigmoid(x_logits) if apply_sigmoid else x_logits
                    x_rec.clamp_(0, 1)
                    
                    val_psnr_sum += psnr_metric(x_rec, x) * x.size(0)
                    val_ssim_sum += ssim_simple(x_rec, x) * x.size(0)
                    val_count += x.size(0)

            total += float(loss.item())
            total_recon += float(recon.item())
            total_kl += float(kl.item())
            steps += 1
            
            postfix = {
                'loss': f"{total/steps:.4f}", 'recon': f"{total_recon/steps:.4f}", 'kl': f"{total_kl/steps:.4f}"
            }
            if not train and val_count > 0:
                postfix.update({'psnr': f"{(val_psnr_sum/val_count):.2f}", 'ssim': f"{(val_ssim_sum/val_count):.4f}"})
            pbar.set_postfix(postfix)

        avg_loss = total / len(loader)
        avg_recon = total_recon / len(loader)
        avg_kl = total_kl / len(loader)
        
        avg_psnr = val_psnr_sum / val_count if val_count > 0 else 0
        avg_ssim = val_ssim_sum / val_count if val_count > 0 else 0

        return avg_loss, avg_recon, avg_kl, global_step, avg_psnr, avg_ssim
    
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
        grad_clip_max_norm: float = 0.0,
        scheduler=None,
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
            train_loss, train_recon, train_kl, global_step, _, _ = self.run_epoch(train_loader, train=True, epoch=epoch, num_epochs=num_epochs, beta=current_beta, grad_clip_max_norm=grad_clip_max_norm, global_step_start=(locals().get('global_step', 0) or 0))
            val_loss, val_recon, val_kl, _, val_psnr, val_ssim = self.run_epoch(val_loader, train=False, epoch=epoch, num_epochs=num_epochs, beta=current_beta, grad_clip_max_norm=0.0, global_step_start=global_step)

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
                    'val_psnr': val_psnr,
                    'val_ssim': val_ssim,
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

            # Step LR scheduler (per-epoch) if provided
            if scheduler is not None:
                scheduler.step()

        if save_latents_flag and output_dir is not None:
            save_spatial_latents(self.model, train_loader, self.device, output_dir / 'latents_train')
            save_spatial_latents(self.model, val_loader, self.device, output_dir / 'latents_val')

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
        from torchvision import transforms as T
        x, _ = next(iter(val_loader))
        x = x.to(self.device)
        with torch.no_grad():
            x_logits, _, _, _ = self.model(x)

            # Decide how to map logits to image space based on loss settings
            if getattr(self.model, 'recon_loss', 'mse') == 'bce' or getattr(self.model, 'mse_use_sigmoid', True):
                x_rec = torch.sigmoid(x_logits)
            else:
                x_rec = x_logits

        # Attempt to detect Normalize(mean, std) in dataset transform and invert it for visualization
        def find_normalize(t):
            if t is None:
                return None
            if isinstance(t, T.Normalize):
                return t
            # torchvison Compose-style container
            sub = getattr(t, 'transforms', None)
            if isinstance(sub, (list, tuple)):
                for s in sub:
                    n = find_normalize(s)
                    if n is not None:
                        return n
            # Fallback: nested attr named 'transform'
            nested = getattr(t, 'transform', None)
            if nested is not None:
                return find_normalize(nested)
            return None

        norm = find_normalize(getattr(val_loader.dataset, 'transform', None))
        def unnormalize(img_batch, normalize_module):
            if normalize_module is None:
                return img_batch
            mean = torch.as_tensor(normalize_module.mean, device=img_batch.device).view(1, -1, 1, 1)
            std = torch.as_tensor(normalize_module.std, device=img_batch.device).view(1, -1, 1, 1)
            return img_batch * std + mean

        x_disp = unnormalize(x, norm).clamp(0, 1)
        x_rec_disp = unnormalize(x_rec, norm).clamp(0, 1)

        grid = vutils.make_grid(torch.cat([x_disp[:8], x_rec_disp[:8]], dim=0), nrow=8)
        img_path = output_dir / 'recon_grid.png'
        torchvision.utils.save_image(grid, img_path)
        if logger is not None:
            logger.log_artifact(img_path)
