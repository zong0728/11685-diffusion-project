#!/bin/bash
#SBATCH --job-name=smoke_lv
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_lv_%j.out
#SBATCH --error=logs/smoke_lv_%j.err

# Quick 2-epoch smoke to confirm learned_range variance + VLB loss works end-to-end.
# If this passes, P3 is safe to launch. If it fails, DO NOT launch P3.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/production/_common.sh
run_train --run_name smoke_learned_var \
          --unet_ch 256 \
          --batch_size 256 \
          --num_epochs 2 \
          --variance_type learned_range \
          --prediction_type epsilon \
          --vlb_weight 0.001 \
          --save_every 1
