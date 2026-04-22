#!/bin/bash
#SBATCH --job-name=eval_P3_24
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/eval_P3_ep24_%j.out
#SBATCH --error=logs/eval_P3_ep24_%j.err

# P3 early checkpoint — FID baseline to see where learned-variance started.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_24.pth \
         --variance_type learned_range \
         --run_name eval_P3_ep24 2>&1 | tee eval_results/P3_ep24.txt
