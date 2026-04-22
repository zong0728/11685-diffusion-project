#!/bin/bash
#SBATCH --job-name=T2_ch384
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=logs/T2_ch384_%j.out
#SBATCH --error=logs/T2_ch384_%j.err

# T2: larger UNet (ch=384, ~360M params) trained from scratch for ~30h.
# Tests whether more model capacity pays off beyond our ch=256 winner.
# batch=256 because bf16 memory budget for ch=384 on one H200.
# ~1.7 min/epoch → ~1000 epochs in 30h.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/production/_common.sh
run_train --run_name T2_ch384 \
          --unet_ch 384 --batch_size 256 --num_epochs 1000 \
          --variance_type learned_range --prediction_type epsilon --vlb_weight 0.001
