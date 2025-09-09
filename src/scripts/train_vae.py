from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import argparse

from src.data import get_data_loaders
from src.models.spatial_vae import SpatialVAE
from src.training.spatial_engine import SpatialTrainingEngine
from src.utils.system import set_seed, get_device
from src.utils.logger import MlflowLogger


def main(config_path: str) -> None:
    with open(config_path, 'r') as f:
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
        'seed': cfg['seed'], 'device': str(device), 'max_epochs': cfg['max_epochs'],
        'lr': cfg['lr'], 'weight_decay': cfg['weight_decay'],
        'latent_dim': cfg['model']['latent_dim'], 'recon_loss': cfg['model']['recon_loss'],
    })

    train_loader, val_loader = get_data_loaders(
        name=str(cfg['data']['name']), root=cfg['data']['root'],
        batch_size=cfg['data']['batch_size'], num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'], persistent_workers=cfg['data']['persistent_workers'],
        augment=bool(cfg['data'].get('augment', False)),
    )

    model = SpatialVAE(**cfg['model']).to(device)

    opt = AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    
    scheduler = CosineAnnealingLR(opt, T_max=int(cfg['max_epochs'])) if cfg.get('scheduler') else None

    engine = SpatialTrainingEngine(model=model, optimizer=opt, device=device)

    dataset_name = str(cfg['data']['name']).lower()
    ds_slug = f"spatial_vae_{dataset_name}"
    out_dir = Path(cfg['out_dir']) / ds_slug
    ckpt_dir = out_dir / 'checkpoints'

    engine.train(
        train_loader=train_loader, val_loader=val_loader,
        num_epochs=cfg['max_epochs'], early_stop=cfg['early_stop'],
        checkpoint_dir=ckpt_dir, logger=logger, output_dir=out_dir,
        save_latents_flag=bool(cfg['save_latents']), beta=float(cfg['model']['beta']),
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
