#!/bin/bash
#SBATCH --job-name=eval_E
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/eval_E_%j.out
#SBATCH --error=logs/eval_E_%j.err

# FID/IS for E_no_cfg (no CFG → unconditional generate path). Same architecture as A.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt experiments/exp-11-E_no_cfg/checkpoints/ema_epoch_143.pth \
         --use_cfg False \
         --run_name eval_E_no_cfg 2>&1 | tee eval_results/E_no_cfg.txt
