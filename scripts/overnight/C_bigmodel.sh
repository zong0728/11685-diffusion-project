#!/bin/bash
#SBATCH --job-name=C_bigmodel
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/C_bigmodel_%j.out
#SBATCH --error=logs/C_bigmodel_%j.err

set -euo pipefail

# Overnight Run C: Bigger UNet (ch=320, ~280M params)
# batch=384 to fit. Tests model capacity scaling.
# 110 epochs target (slower per-epoch than baseline)

module load python/miniforge3_pytorch/2.10.0
export PYTHONUSERBASE=$HOME/.local
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

cd /projects/bgyq/sguan/11685-diffusion-project
mkdir -p logs experiments
export WANDB_MODE=offline

python train.py \
    --config configs/ddpm.yaml \
    --data_dir /work/nvme/bgyq/sguan/imagenet100_128x128/train \
    --run_name C_bigmodel \
    --image_size 128 --unet_in_size 32 --unet_in_ch 3 \
    --unet_ch 320 --unet_ch_mult 1 2 3 4 --unet_attn 2 3 --unet_num_res_blocks 2 --unet_dropout 0.0 \
    --num_epochs 110 --batch_size 384 --num_workers 16 --num_classes 100 \
    --learning_rate 1e-4 --weight_decay 1e-4 \
    --num_train_timesteps 1000 --num_inference_steps 50 --beta_schedule cosine \
    --latent_ddpm True --use_cfg True --cfg_guidance_scale 2.0 --use_ddim True \
    --use_ema True --ema_decay 0.9999 \
    --mixed_precision bf16 --min_snr_gamma 5.0 \
    --prediction_type epsilon
