#!/bin/bash
#SBATCH --job-name=eval_D
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/eval_D_%j.out
#SBATCH --error=logs/eval_D_%j.err

# FID/IS for D_no_minsnr (γ=0). Same architecture as A.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt experiments/exp-12-D_no_minsnr/checkpoints/ema_epoch_138.pth \
         --run_name eval_D_no_minsnr 2>&1 | tee eval_results/D_no_minsnr.txt
