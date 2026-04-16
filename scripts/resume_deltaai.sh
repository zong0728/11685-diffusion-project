#!/bin/bash
#SBATCH --job-name=ddpm_resume
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err

set -euo pipefail

# Resume training from the last checkpoint.
# Usage: sbatch scripts/resume_deltaai.sh <ckpt_path>
# Example:
#   sbatch scripts/resume_deltaai.sh experiments/exp-0-latent_cfg_v1/checkpoints/checkpoint_epoch_99.pth

CKPT_PATH="${1:?Please provide checkpoint path as first argument}"

echo "Job starting at $(date)"
echo "Running on $(hostname)"
echo "Resuming from: $CKPT_PATH"
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
    --run_name latent_cfg_v1 \
    --image_size 128 \
    --unet_in_size 32 \
    --unet_ch 192 \
    --unet_ch_mult 1 2 2 4 \
    --unet_attn 2 3 \
    --num_epochs 500 \
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
    --ema_decay 0.9999 \
    --resume "$CKPT_PATH"

echo "Job finished at $(date)"
