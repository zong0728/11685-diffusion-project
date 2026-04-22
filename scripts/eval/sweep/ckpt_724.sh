#!/bin/bash
#SBATCH --job-name=ckpt_724
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ckpt_724_%j.out
#SBATCH --error=logs/ckpt_724_%j.err

source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_724.pth \
         --variance_type learned_range \
         --run_name sweep_ep724 2>&1 | tee eval_results/sweep_ep724.txt
