from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path

from src.data.sequences import SequenceDataset, SequenceDataConfig, pad_collate_fn, SPECIAL_TOKENS
from src.models.transformer import GPTTransformer, TransformerConfig
from src.utils.system import set_seed, get_device
from src.utils.logger import MlflowLogger


def compute_ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    # logits: [B, T, V], labels: [B, T]
    B, T, V = logits.size()
    return torch.nn.functional.cross_entropy(
        logits.view(B * T, V), labels.view(B * T), ignore_index=ignore_index
    )


@hydra.main(version_base=None, config_path='../../configs', config_name='train_transformer')
def main(cfg) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    # Load data config (dataset-agnostic, points to codes and labels files)
    data_cfg = OmegaConf.load(to_absolute_path(cfg.data_config_path))
    model_cfg = OmegaConf.load(to_absolute_path(cfg.model_config_path))

    # Logger
    logger = MlflowLogger(
        tracking_uri=to_absolute_path(cfg.mlflow_tracking_uri),
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
    )

    # Dataset
    seq_cfg = SequenceDataConfig(
        codes_path=to_absolute_path(data_cfg.codes_path),
        labels_path=(to_absolute_path(data_cfg.labels_path) if 'labels_path' in data_cfg else None),
        add_bos=bool(getattr(data_cfg, 'add_bos', True)),
        add_eos=bool(getattr(data_cfg, 'add_eos', True)),
        add_cls=bool(getattr(data_cfg, 'add_cls', False)),
        pad_token_id=int(getattr(data_cfg, 'pad_token_id', SPECIAL_TOKENS['PAD'])),
        bos_token_id=int(getattr(data_cfg, 'bos_token_id', SPECIAL_TOKENS['BOS'])),
        eos_token_id=int(getattr(data_cfg, 'eos_token_id', SPECIAL_TOKENS['EOS'])),
        cls_token_id=int(getattr(data_cfg, 'cls_token_id', SPECIAL_TOKENS['CLS'])),
    )
    ds = SequenceDataset(seq_cfg)
    # Simple split (90/10)
    n_total = len(ds)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=data_cfg.batch_size, shuffle=True,
        num_workers=data_cfg.num_workers, pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.persistent_workers,
        collate_fn=lambda b: pad_collate_fn(b, pad_token_id=seq_cfg.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg.batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.persistent_workers,
        collate_fn=lambda b: pad_collate_fn(b, pad_token_id=seq_cfg.pad_token_id),
    )

    # Model
    grid_size = None
    if bool(getattr(model_cfg, 'use_2d_positions', False)):
        grid_size = (int(model_cfg.grid_h), int(model_cfg.grid_w))
    tcfg = TransformerConfig(
        vocab_size=int(model_cfg.vocab_size),
        max_seq_len=int(model_cfg.max_seq_len),
        d_model=int(model_cfg.d_model),
        n_heads=int(model_cfg.n_heads),
        n_layers=int(model_cfg.n_layers),
        dropout=float(model_cfg.dropout),
        use_2d_positions=bool(getattr(model_cfg, 'use_2d_positions', False)),
        grid_size=grid_size,
        num_classes=int(model_cfg.num_classes) if 'num_classes' in model_cfg and model_cfg.num_classes is not None else None,
    )
    model = GPTTransformer(tcfg).to(device)

    # Optimizer
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Training loop
    ckpt_dir = Path(to_absolute_path(cfg.ckpt_dir))
    out_dir = Path(to_absolute_path(cfg.out_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.log_params({
        **asdict(tcfg),
        'seed': cfg.seed,
        'device': str(device),
        'max_epochs': cfg.max_epochs,
        'lr': cfg.lr,
        'weight_decay': cfg.weight_decay,
        'batch_size': data_cfg.batch_size,
    })

    ignore_index = seq_cfg.pad_token_id

    def run_epoch(loader, train: bool, epoch: int):
        model.train(train)
        total, steps = 0.0, 0
        for batch in loader:
            input_ids, attention_mask, class_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if class_labels is not None:
                class_labels = class_labels.to(device)

            # Teacher forcing: predict next token
            logits = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1], class_labels=class_labels)
            labels = input_ids[:, 1:].contiguous()
            loss = compute_ce_loss(logits, labels, ignore_index=ignore_index)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            total += float(loss.item())
            steps += 1
        return total / max(1, steps)

    best_val = float('inf')
    no_improve = 0
    for epoch in range(1, cfg.max_epochs + 1):
        print(f"Epoch {epoch}/{cfg.max_epochs}")
        tr = run_epoch(train_loader, True, epoch)
        va = run_epoch(val_loader, False, epoch)
        logger.log_metrics({'train_loss': tr, 'val_loss': va}, step=epoch)
        if va < best_val:
            best_val = va
            no_improve = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch}, ckpt_dir / 'best.pt')
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    logger.end()
    print("Done. Artifacts in:", out_dir)


if __name__ == '__main__':
    main()

