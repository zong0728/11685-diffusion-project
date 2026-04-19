#!/bin/bash
#SBATCH --job-name=P3_learnedvar
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=47:30:00
#SBATCH --output=logs/P3_learnedvar_%j.out
#SBATCH --error=logs/P3_learnedvar_%j.err

# P3: Improved-DDPM-style learned variance (Nichol & Dhariwal 2021). Uses ε-prediction
# (not v-prediction) because the hybrid L_simple + λ·L_vlb objective is paired with ε
# in the original paper, and mixing v-pred with learned_range is untested territory.
# Everything else matches the winning overnight config: cosine + CFG=2 + min-SNR=5 + EMA.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/production/_common.sh
run_train --run_name P3_learned_var \
          --unet_ch 256 \
          --batch_size 512 \
          --num_epochs 780 \
          --variance_type learned_range \
          --prediction_type epsilon \
          --vlb_weight 0.001
