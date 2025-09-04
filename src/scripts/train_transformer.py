import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import hydra
from hydra.utils import to_absolute_path

from src.data import get_code_loaders
from src.models import Transformer
from src.utils.system import get_device, set_seed


@hydra.main(version_base=None, config_path='../../configs', config_name='presets/fashion_spatial_geodesic/3_train_transformer')
def main(cfg):
    """Main training loop for the autoregressive Transformer."""
    set_seed(cfg.system.seed)
    device = get_device(cfg.system.device)

    data_cfg = cfg.data
    train_loader, val_loader = get_code_loaders(
        codes_path=to_absolute_path(data_cfg.codes_path),
        labels_path=to_absolute_path(data_cfg.labels_path) if data_cfg.labels_path else None,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
    )

    model_cfg = cfg.model
    model = Transformer(
        num_classes=model_cfg.num_classes,
        num_tokens=model_cfg.num_tokens,
        embed_dim=model_cfg.embed_dim,
        n_layers=model_cfg.n_layers,
        n_head=model_cfg.n_head,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout,
    ).to(device)

    train_cfg = cfg.training
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(train_cfg.epochs))

    out_dir = Path(to_absolute_path(cfg.out.dir))
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(train_cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']} [Train]")
        total_loss = 0

        for batch in pbar:
            x, y = batch[0].to(device), batch[1].to(device)
            labels = batch[2].to(device) if model.num_classes > 0 else None
            
            logits = model(x, y=labels)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                labels = batch[2].to(device) if model.num_classes > 0 else None
                logits = model(x, y=labels)
                val_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        torch.save(model.state_dict(), ckpt_dir / "latest.pt")


if __name__ == "__main__":
    main() 