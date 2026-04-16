#!/bin/bash
#SBATCH --job-name=ddpm_smoke
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail

# =========================================================
# Smoke test: 2 epochs of latent DDPM on DeltaAI
# Purpose: verify GPU, data loading, VAE, EMA, cosine schedule all work
# Cost: ~10-20 min on one GH200 → spends ~0.5 GPU-hours
# =========================================================

echo "Job starting at $(date)"
echo "Running on $(hostname)"
nvidia-smi || true

module purge
module load python/miniforge3_pytorch/2.10.0

export PYTHONUSERBASE=$HOME/.local
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

PROJECT_DIR=/projects/bgyq/sguan/11685-diffusion-project
DATA_DIR=/work/nvme/bgyq/sguan/imagenet100_128x128

cd "$PROJECT_DIR"
mkdir -p logs experiments

export WANDB_MODE=offline

python train.py \
    --config configs/ddpm.yaml \
    --data_dir "$DATA_DIR/train" \
    --run_name smoke_test \
    --image_size 128 \
    --unet_in_size 32 \
    --unet_in_ch 3 \
    --unet_ch 128 \
    --unet_ch_mult 1 2 2 4 \
    --unet_attn 2 3 \
    --unet_num_res_blocks 2 \
    --num_epochs 2 \
    --batch_size 64 \
    --num_workers 8 \
    --num_classes 100 \
    --learning_rate 1e-4 \
    --num_train_timesteps 1000 \
    --num_inference_steps 50 \
    --beta_schedule cosine \
    --latent_ddpm True \
    --use_cfg True \
    --cfg_guidance_scale 2.0 \
    --use_ddim True \
    --use_ema True \
    --ema_decay 0.9999

echo "Smoke test finished at $(date)"
echo "If you see this message without errors, the full pipeline works end-to-end on a GPU."
