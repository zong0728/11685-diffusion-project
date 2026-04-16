#!/bin/bash
#SBATCH --job-name=ddpm_bench4
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench4_%j.out
#SBATCH --error=logs/bench4_%j.err

set -euo pipefail

# =========================================================
# Benchmark v4: UNet ch 320, batch 384 (larger model)
# Purpose: see if a bigger model + smaller batch is faster per-sample
# =========================================================

echo "Job starting at $(date)"
echo "Running on $(hostname)"
nvidia-smi || true

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
    --run_name bench_v4_ch320_b384 \
    --image_size 128 \
    --unet_in_size 32 \
    --unet_in_ch 3 \
    --unet_ch 320 \
    --unet_ch_mult 1 2 3 4 \
    --unet_attn 2 3 \
    --unet_num_res_blocks 2 \
    --unet_dropout 0.0 \
    --num_epochs 1 \
    --batch_size 384 \
    --num_workers 16 \
    --num_classes 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --num_train_timesteps 1000 \
    --num_inference_steps 50 \
    --beta_schedule cosine \
    --latent_ddpm True \
    --use_cfg True \
    --cfg_guidance_scale 2.0 \
    --use_ddim True \
    --use_ema True \
    --ema_decay 0.9999 \
    --mixed_precision bf16 \
    --min_snr_gamma 5.0

echo "Benchmark v4 finished at $(date)"
