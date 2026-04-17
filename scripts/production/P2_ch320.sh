#!/bin/bash
#SBATCH --job-name=P2_ch320
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=47:30:00
#SBATCH --output=logs/P2_ch320_%j.out
#SBATCH --error=logs/P2_ch320_%j.err

# P2 (upside): B+C combo. ch=320 + v-pred + batch=384, ~4.4 min/epoch
# → ~640 epochs in 47.5h. Higher ceiling but v-pred × bigmodel unvalidated.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/production/_common.sh
run_train --run_name P2_ch320 \
          --unet_ch 320 \
          --batch_size 384 \
          --num_epochs 640
