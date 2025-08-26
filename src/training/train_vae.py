from pathlib import Path

from torch.optim import Adam
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path

from src.data import get_data_loaders
from src.models.vae import VAE
from src.training.engine import TrainingEngine
from src.utils.system import set_seed, get_device
from src.utils.logger import MlflowLogger


@hydra.main(version_base=None, config_path='../../configs', config_name='train')
def main(cfg) -> None:
    # Hydra changes CWD; use absolute paths for config files
    # Load root-level configs only
    data_cfg = OmegaConf.load(to_absolute_path('configs/data.yaml'))
    vae_cfg  = OmegaConf.load(to_absolute_path('configs/vae.yaml'))

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    # Logger (MLflow)
    logger = MlflowLogger(
        tracking_uri=to_absolute_path(cfg.mlflow_tracking_uri),
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
    )
    logger.log_params({
        'seed': cfg.seed,
        'device': str(device),
        'max_epochs': cfg.max_epochs,
        'lr': cfg.lr,
        'weight_decay': cfg.weight_decay,
        'latent_dim': vae_cfg.latent_dim,
        'recon_loss': vae_cfg.recon_loss,
    })

    # Data
    train_loader, val_loader = get_data_loaders(
        name=str(getattr(data_cfg, 'name', 'MNIST')),
        root=to_absolute_path(data_cfg.root),
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.persistent_workers,
        augment=bool(getattr(data_cfg, 'augment', False)),
    )

    # Model
    model = VAE(
        in_channels=getattr(vae_cfg, 'in_channels', 1),
        enc_channels=vae_cfg.enc_channels,
        dec_channels=vae_cfg.dec_channels,
        latent_dim=vae_cfg.latent_dim,
        recon_loss=vae_cfg.recon_loss,
        output_image_size=int(getattr(vae_cfg, 'output_image_size', 28)),
    ).to(device)

    # Optimizer
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Train via engine
    engine = TrainingEngine(
        model=model,
        optimizer=opt,
        device=device,
    )

    # Auto-adjust directories by dataset slug to keep results clustered per dataset
    dataset_name = str(getattr(data_cfg, 'name', 'MNIST')).strip().lower()
    ds_slug = 'vae_' + ('cifar10' if dataset_name == 'cifar10' else 'fashion' if 'fashion' in dataset_name else 'mnist')
    base_out = Path(to_absolute_path(str(cfg.out_dir)))
    out_dir = base_out.parent / ds_slug
    ckpt_dir = out_dir / 'checkpoints'

    engine.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.max_epochs,
        early_stop=cfg.early_stop,
        checkpoint_dir=ckpt_dir,
        logger=logger,
        output_dir=out_dir,
        save_latents_flag=bool(cfg.save_latents),
        kl_anneal_epochs=int(getattr(cfg, 'kl_anneal_epochs', 0)),
        beta=float(getattr(vae_cfg, 'beta', 1.0)),
    )

    logger.end()
    print("Done. Artifacts in:", out_dir)

if __name__ == '__main__':
    main()