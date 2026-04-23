#!/bin/bash
#SBATCH --job-name=T5_sigmoid
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/T5_sigmoid_%j.out
#SBATCH --error=logs/T5_sigmoid_%j.err

# E9 / T5: sigmoid noise schedule (Chen 2023). Heavier weight on intermediate
# noise levels — claimed to help at 128×128 resolution. TA encouraged
# "explore noise schedules" in the project PPT, this is the under-explored one.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/production/_common.sh
run_train --run_name T5_sigmoid \
          --unet_ch 256 \
          --batch_size 512 \
          --num_epochs 390 \
          --variance_type learned_range \
          --prediction_type epsilon \
          --vlb_weight 0.001 \
          --beta_schedule sigmoid
