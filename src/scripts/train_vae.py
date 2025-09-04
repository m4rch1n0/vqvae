from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path

from src.data import get_data_loaders
from src.models.spatial_vae import SpatialVAE
from src.training.engine import TrainingEngine
from src.utils.system import set_seed, get_device
from src.utils.logger import MlflowLogger


@hydra.main(version_base=None, config_path='../../configs', config_name='presets/fashion_spatial_geodesic/1_train_vae')
def main(cfg) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    logger = MlflowLogger(
        tracking_uri=to_absolute_path(cfg.mlflow_tracking_uri),
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
    )
    logger.log_params({
        'seed': cfg.seed, 'device': str(device), 'max_epochs': cfg.max_epochs,
        'lr': cfg.lr, 'weight_decay': cfg.weight_decay,
        'latent_dim': cfg.model.latent_dim, 'recon_loss': cfg.model.recon_loss,
    })

    train_loader, val_loader = get_data_loaders(
        name=str(cfg.data.name), root=to_absolute_path(cfg.data.root),
        batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory, persistent_workers=cfg.data.persistent_workers,
        augment=bool(getattr(cfg.data, 'augment', False)),
    )

    model = SpatialVAE(
        in_channels=cfg.model.in_channels, enc_channels=cfg.model.enc_channels,
        dec_channels=cfg.model.dec_channels, latent_dim=cfg.model.latent_dim,
        recon_loss=cfg.model.recon_loss, output_image_size=cfg.model.output_image_size,
        norm_type=cfg.model.norm_type, mse_use_sigmoid=bool(getattr(cfg.model, 'mse_use_sigmoid', True)),
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    scheduler = CosineAnnealingLR(opt, T_max=cfg.max_epochs) if getattr(cfg, 'scheduler', None) else None

    engine = TrainingEngine(model=model, optimizer=opt, device=device)

    dataset_name = str(cfg.data.name).lower()
    ds_slug = f"spatial_vae_{dataset_name}"
    out_dir = Path(to_absolute_path(str(cfg.out_dir))) / ds_slug
    ckpt_dir = out_dir / 'checkpoints'

    # The `train` method from the original engine should be compatible
    engine.train(
        train_loader=train_loader, val_loader=val_loader,
        num_epochs=cfg.max_epochs, early_stop=cfg.early_stop,
        checkpoint_dir=ckpt_dir, logger=logger, output_dir=out_dir,
        save_latents_flag=bool(cfg.save_latents), beta=float(cfg.model.beta),
        grad_clip_max_norm=float(getattr(cfg, 'grad_clip_max_norm', 0.0)),
        scheduler=scheduler,
    )

    logger.end()
    print("Done. Artifacts in:", out_dir)

if __name__ == '__main__':
    main()
