#!/bin/bash
#SBATCH --job-name=T4_linear
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/T4_linear_%j.out
#SBATCH --error=logs/T4_linear_%j.err

# E6 / T4: linear noise schedule (vs cosine in P3) — direct ablation for the report.
# Same architecture and CFG as P3, only --beta_schedule changes.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/production/_common.sh
run_train --run_name T4_linear \
          --unet_ch 256 \
          --batch_size 512 \
          --num_epochs 390 \
          --variance_type learned_range \
          --prediction_type epsilon \
          --vlb_weight 0.001 \
          --beta_schedule linear
