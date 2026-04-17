#!/bin/bash
#SBATCH --job-name=P1_ch256
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=47:30:00
#SBATCH --output=logs/P1_ch256_%j.out
#SBATCH --error=logs/P1_ch256_%j.err

# P1 (safe): B_vpred winner scaled up. ch=256, batch=512, ~3.6 min/epoch
# → ~780 epochs in 47.5h. Target FID <30.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/production/_common.sh
run_train --run_name P1_ch256 \
          --unet_ch 256 \
          --batch_size 512 \
          --num_epochs 780
