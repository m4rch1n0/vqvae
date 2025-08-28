from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path

from src.data.sequence_datasets import CodeSequenceDataset, SpecialTokens, pad_collate
from src.models.transformer import GPT, GPTConfig
from src.utils.system import set_seed, get_device
from src.utils.logger import MlflowLogger


@hydra.main(version_base=None, config_path='../../configs/transformer', config_name='cifar10_small')
def main(cfg) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    # Load additional root-level configs for dataset-specific paths if needed
    data_root_cfg = OmegaConf.load(to_absolute_path('configs/data.yaml'))

    # Resolve absolute paths
    codes_train = Path(to_absolute_path(cfg.data.codes_train))
    codes_val = Path(to_absolute_path(cfg.data.codes_val))
    classes_train = Path(to_absolute_path(cfg.data.classes_train)) if cfg.data.get('classes_train') else None
    classes_val = Path(to_absolute_path(cfg.data.classes_val)) if cfg.data.get('classes_val') else None

    # Datasets
    ds_train = CodeSequenceDataset(
        codes_path=codes_train,
        classes_path=classes_train,
        add_bos=cfg.tokens.add_bos,
        add_eos=cfg.tokens.add_eos,
        special_tokens=None,
    )
    ds_val = CodeSequenceDataset(
        codes_path=codes_val,
        classes_path=classes_val,
        add_bos=cfg.tokens.add_bos,
        add_eos=cfg.tokens.add_eos,
        special_tokens=None,
    )

    # Auto-assign compact special token IDs after base vocab to avoid huge embeddings
    base_vocab = max(ds_train.base_vocab_size, ds_val.base_vocab_size)
    pad_id = base_vocab
    bos_id = (base_vocab + 1) if bool(cfg.tokens.add_bos) else None
    eos_id = (base_vocab + 2) if bool(cfg.tokens.add_eos) else None
    specials = SpecialTokens(bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
    ds_train.special_tokens = specials
    ds_val.special_tokens = specials

    # Vocab size (with specials)
    vocab_size = max(ds_train.vocab_size_with_specials(), ds_val.vocab_size_with_specials())
    seq_len = int(cfg.model.seq_len)

    # Model
    model_cfg = GPTConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=cfg.model.embed_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        attn_dropout=cfg.model.attn_dropout,
        resid_dropout=cfg.model.resid_dropout,
        mlp_ratio=cfg.model.mlp_ratio,
        class_conditional=bool(cfg.model.class_conditional),
        num_classes=int(cfg.model.num_classes) if cfg.model.get('num_classes') else 0,
    )
    model = GPT(model_cfg).to(device)

    # DataLoaders
    collate = lambda batch: pad_collate(batch, pad_id=pad_id)
    train_loader = DataLoader(ds_train, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, collate_fn=collate)
    val_loader = DataLoader(ds_val, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=collate)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    # Logger
    logger = MlflowLogger(
        tracking_uri=to_absolute_path(cfg.mlflow_tracking_uri),
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
    )
    logger.log_params({
        'seed': cfg.seed,
        'device': str(device),
        'lr': cfg.training.lr,
        'batch_size': cfg.training.batch_size,
        'embed_dim': cfg.model.embed_dim,
        'num_layers': cfg.model.num_layers,
        'num_heads': cfg.model.num_heads,
        'seq_len': seq_len,
        'vocab_size': vocab_size,
    })

    # Training loop
    best_val = float('inf')
    out_dir = Path(to_absolute_path(cfg.out.dir))
    ckpt_dir = out_dir / 'checkpoints'
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.training.max_epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            classes = batch.get('class_id')
            classes = classes.to(device) if classes is not None else None
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, labels=labels, classes=classes)
            loss.backward()
            if cfg.training.grad_clip_max_norm and cfg.training.grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.training.grad_clip_max_norm))
            optimizer.step()
            total += float(loss.item())
            steps += 1
        train_loss = total / max(1, steps)

        # Validation
        model.eval()
        total_v = 0.0
        steps_v = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                classes = batch.get('class_id')
                classes = classes.to(device) if classes is not None else None
                logits, loss = model(input_ids, labels=labels, classes=classes)
                total_v += float(loss.item())
                steps_v += 1
        val_loss = total_v / max(1, steps_v)

        logger.log_metrics({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)
        print(f"Epoch {epoch}/{cfg.training.max_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, ckpt_dir / 'best.pt')

    torch.save({'model_state_dict': model.state_dict(), 'epoch': cfg.training.max_epochs}, ckpt_dir / 'latest.pt')
    logger.end()
    print("Done. Artifacts in:", out_dir)


if __name__ == '__main__':
    main()


