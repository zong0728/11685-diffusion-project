#!/bin/bash
#SBATCH --job-name=ddpm_train
#SBATCH --account=cis260706      # Your ACCESS project ID
#SBATCH --partition=ghx4         # DeltaAI GPU partition — confirm name in docs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1        # Request 1 H200 GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00          # 48 hours — max allowed per job
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

# =========================================================
# NCSA DeltaAI — Latent DDPM training job
# =========================================================

echo "Job starting at $(date)"
echo "Running on $(hostname)"
nvidia-smi || true

# Load modules (adjust to DeltaAI's actual module names; check `module avail`)
module purge
module load cuda/12.4 || true
module load anaconda3 || true

# Activate your conda environment (create once ahead of time)
source activate diffusion || conda activate diffusion

# Move into project directory (edit PROJECT_DIR to your actual path)
cd $PROJECT_DIR/11685-diffusion-project

# Make sure logs/ and experiments/ exist
mkdir -p logs experiments

# =========================================================
# Training command
# =========================================================
# Latent DDPM + CFG + big UNet + cosine schedule + EMA
python train.py \
    --config configs/ddpm.yaml \
    --data_dir "$DATA_DIR/imagenet100_128x128/train" \
    --run_name latent_cfg_v1 \
    --image_size 128 \
    --unet_in_size 32 \
    --unet_in_ch 3 \
    --unet_ch 192 \
    --unet_ch_mult 1 2 2 4 \
    --unet_attn 2 3 \
    --unet_num_res_blocks 2 \
    --unet_dropout 0.0 \
    --num_epochs 500 \
    --batch_size 64 \
    --num_workers 8 \
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
    --ema_decay 0.9999

echo "Job finished at $(date)"
