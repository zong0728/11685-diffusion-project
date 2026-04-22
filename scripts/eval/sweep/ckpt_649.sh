#!/bin/bash
#SBATCH --job-name=ckpt_649
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ckpt_649_%j.out
#SBATCH --error=logs/ckpt_649_%j.err

# S3: fill the gap between ep524 (FID 59.9) and ep779 (FID 64.5) — see if late-mid is best.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_649.pth \
         --variance_type learned_range \
         --run_name sweep_ep649 2>&1 | tee eval_results/sweep_ep649.txt
