#!/bin/bash
#SBATCH --job-name=eval_B
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/eval_B_%j.out
#SBATCH --error=logs/eval_B_%j.err

# FID/IS for B_vpred (v-prediction). Same architecture as A.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt experiments/exp-14-B_vpred/checkpoints/ema_epoch_133.pth \
         --prediction_type v_prediction \
         --run_name eval_B_vpred 2>&1 | tee eval_results/B_vpred.txt
