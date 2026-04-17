#!/bin/bash
#SBATCH --job-name=eval_A
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/eval_A_%j.out
#SBATCH --error=logs/eval_A_%j.err

# FID/IS for A_baseline (epsilon, min-SNR=5, CFG=2.0, ch=256), latest ema.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt experiments/exp-15-A_baseline/checkpoints/ema_epoch_134.pth \
         --run_name eval_A_baseline 2>&1 | tee eval_results/A_baseline.txt
