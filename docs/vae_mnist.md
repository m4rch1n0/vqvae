geo-vqvae/
  configs/
    data.yaml          # Iperparametri dataloader (batch_size, num_workers, pin_memory, augment)
    vae.yaml           # Architettura VAE e iperparametri (latent_dim, β, canali, tipo recon_loss)
    train.yaml         # Parametri training (seed, epochs, lr, device, cartelle out)
  src/
    data/
      mnist.py         # get_mnist_loaders(): dataset MNIST + DataLoader ottimizzati (workers/pin_memory)
    models/
      vae.py           # Encoder, Decoder, classe VAE, funzione loss (ELBO = recon + β*KL)
    training/
      train_vae.py     # Script di training con Hydra, MLflow, checkpoint e salvataggi
  scripts/
    setup_env.sh       # Verifica ROCm, stampa info torch/hip, installa requirements
    download_data.sh   # Scarica MNIST
    run_train_vae.sh   # Avvio training con override Hydra (epochs, lr, out_dir ecc.)
  experiments/         # Output di run: mlruns, checkpoint, latenti, immagini ricostruite
  requirements.txt     # Dipendenze (torch rocm, torchvision, hydra-core/omegaconf, mlflow, tqdm, numpy, scipy)
  README.md            # Istruzioni d’uso
