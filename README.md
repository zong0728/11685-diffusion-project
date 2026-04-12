# 11-685 Guided Project: Latent Denoising Diffusion Probabilistic Models

**Author:** Shengzhong Guan (`shengzhg@andrew.cmu.edu`)
**Course:** 11-685 Introduction to Deep Learning, Spring 2026 (Carnegie Mellon University)

This repository contains my implementation for the guided project of 11-685. The project explores Denoising Diffusion Probabilistic Models (DDPM) and their extensions, trained on the 128×128 ImageNet-100 dataset. The implementation covers the four required components:

1. **DDPM** implemented from scratch (U-Net, noise scheduler, training pipeline).
2. **DDIM** sampler for accelerated, deterministic inference.
3. **Latent DDPM** leveraging a pre-trained VAE to perform diffusion in latent space.
4. **Classifier-Free Guidance (CFG)** for class-conditional image generation.

## Repository Structure

```
hw5_student_starter_code/
├── configs/
│   └── ddpm.yaml               # Training hyperparameters
├── models/
│   ├── unet.py                 # U-Net backbone
│   ├── unet_modules.py         # ResBlock, TimeEmbedding, Attention
│   ├── vae.py                  # VAE encode/decode wrappers
│   ├── vae_modules.py          # VAE Encoder/Decoder internals
│   ├── vae_distributions.py    # Diagonal Gaussian posterior
│   └── class_embedder.py       # CFG class conditioning
├── schedulers/
│   ├── scheduling_ddpm.py      # DDPM noise scheduler
│   └── scheduling_ddim.py      # DDIM scheduler
├── pipelines/
│   └── ddpm.py                 # Inference pipeline (DDPM/DDIM + VAE + CFG)
├── utils/                      # Checkpoint, distributed, metric helpers
├── train.py                    # Training entry point
├── inference.py                # Evaluation (FID / IS)
├── generate_submission.py      # Kaggle CSV generator
└── fid_utils.py                # Inception feature utilities
```

## Usage

### Training

```bash
python train.py --config configs/ddpm.yaml
```

Key hyperparameters can be overridden via CLI, e.g.:

```bash
python train.py --config configs/ddpm.yaml \
    --num_epochs 100 \
    --batch_size 128 \
    --learning_rate 1e-4
```

### Inference and Evaluation

```bash
python inference.py --config configs/ddpm.yaml \
    --ckpt experiments/<run_name>/checkpoints/checkpoint_epoch_<N>.pth \
    --use_ddim True
```

This generates 5,000 images and computes FID and Inception Score against the validation set.

### Data

The 128×128 ImageNet-100 dataset (~130k images across 100 classes) is expected under `data/imagenet100_128x128/`. The pre-trained VAE checkpoint (for Latent DDPM) belongs in `pretrained/model.ckpt`.

## Current Progress (Midterm)

- Basic DDPM training pipeline is fully functional.
- DDIM sampler integrated for inference-time acceleration.
- Preliminary evaluation on a 15-epoch 64×64 configuration yields FID ≈ 97.24 and IS ≈ 5.30.
- Latent DDPM and CFG modules are implemented and undergoing training.

## References

- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021.
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022.
- Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS Workshop 2021.
