#!/bin/bash
#SBATCH --job-name=T3_vpred_lv
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/T3_vpred_lv_%j.out
#SBATCH --error=logs/T3_vpred_lv_%j.err

# E5 / T3: v-prediction × learned_range variance, otherwise identical to P3.
# Untested combination — Improved DDPM (learned variance) was paired with ε,
# but Salimans 2022 v-pred was paired with fixed variance. We're curious whether
# the gains stack or interfere. 24h budget => ~390 epochs at batch=512.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/production/_common.sh
run_train --run_name T3_vpred_lv \
          --unet_ch 256 \
          --batch_size 512 \
          --num_epochs 390 \
          --variance_type learned_range \
          --prediction_type v_prediction \
          --vlb_weight 0.001
