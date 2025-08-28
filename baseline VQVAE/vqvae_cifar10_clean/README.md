# VQ-VAE (EMA) – CIFAR-10 Clean Baseline

Baseline **VQ-VAE con codebook EMA** su **CIFAR-10**, pensato per confronti stabili.
- Ricostruzioni nitide (L1 loss), codebook EMA per stabilità.
- Configurabile via `config.yaml`.
- Salva grid di ricostruzioni e checkpoint.
- Nessuna dipendenza extra (solo torch/torchvision, pyyaml, tqdm).

## Installazione (CPU/CUDA/ROCm)
Consiglio un venv:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
> Se usi ROCm (AMD), installa la build corretta di torch seguendo le istruzioni ufficiali, ad es.:
> https://pytorch.org/get-started/locally/ (seleziona ROCm) e poi `pip install torchvision` compatibile.

## Esecuzione
```bash
python train.py --config config.yaml
```
Parametri CLI sovrascrivono il file di config (es. `--epochs 50 --batch_size 256`).

## Output
- `outputs/recon_epochXXXX.png`: griglie di ricostruzioni
- `outputs/checkpoints/ckpt_last.pt`: ultimo checkpoint
- `outputs/checkpoints/ckpt_best.pt`: migliore (per loss totale)
- `outputs/log.csv`: log di training

## Note
- Questo baseline è intenzionalmente semplice per evitare artefatti “verdi”: L1 loss, normalizzazione simmetrica [-1,1], Tanh nel decoder, codebook EMA.
- Per confronto con varianti (geodesic/VQVAE-2), mantieni stessi preprocess e loss per isolamento delle variabili.
