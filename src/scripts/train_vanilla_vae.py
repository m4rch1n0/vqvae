"""Train Vanilla VAE with configurable parameters"""

from pathlib import Path
import yaml
import argparse
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import get_data_loaders
from src.models.vae import VAE
from src.training.engine import TrainingEngine
from src.utils.system import set_seed, get_device
from src.utils.logger import MlflowLogger


def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['seed'])
    device = get_device(cfg['device'])
    print(f"Using device: {device}")

    logger = MlflowLogger(
        tracking_uri=cfg['mlflow_tracking_uri'],
        experiment_name=cfg['experiment_name'],
        run_name=cfg['run_name'],
    )
    logger.log_params({
        'seed': cfg['seed'],
        'device': str(device),
        'max_epochs': cfg['max_epochs'],
        'lr': cfg['lr'],
        'weight_decay': cfg['weight_decay'],
        'latent_dim': cfg['model']['latent_dim'],
        'recon_loss': cfg['model']['recon_loss'],
    })

    # Data
    data_cfg = cfg['data']
    train_loader, val_loader = get_data_loaders(
        name=str(data_cfg['name']),
        root=data_cfg['root'],
        batch_size=data_cfg['batch_size'],
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
        persistent_workers=data_cfg['persistent_workers'],
        augment=bool(data_cfg.get('augment', False)),
    )

    # Model - Vanilla VAE
    model = VAE(**cfg['model']).to(device)

    # Optimizer
    optimizer_class = AdamW if cfg.get('optimizer', 'adamw') == 'adamw' else Adam
    opt = optimizer_class(
        model.parameters(), 
        lr=float(cfg['lr']), 
        weight_decay=float(cfg['weight_decay'])
    )

    # Scheduler
    scheduler = None
    if cfg.get('scheduler') and cfg['scheduler'].get('name') == 'cosine':
        scheduler = CosineAnnealingLR(opt, T_max=int(cfg['max_epochs']))

    # Training engine
    engine = TrainingEngine(model=model, optimizer=opt, device=device)

    # Output directory
    out_dir = Path(cfg['out_dir'])
    ckpt_dir = out_dir / 'checkpoints'

    # Train
    engine.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(cfg['max_epochs']),
        early_stop=int(cfg.get('early_stop', 0)),
        checkpoint_dir=ckpt_dir,
        logger=logger,
        output_dir=out_dir,
        save_latents_flag=bool(cfg.get('save_latents', True)),
        beta=float(cfg.get('beta', 1.0)),
        grad_clip_max_norm=float(cfg.get('grad_clip_max_norm', 0.0)),
        scheduler=scheduler,
    )

    logger.end()
    print("Done. Artifacts in:", out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the training config file.")
    args = parser.parse_args()
    main(args.config)

