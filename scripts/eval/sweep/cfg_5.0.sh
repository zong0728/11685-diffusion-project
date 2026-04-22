#!/bin/bash
#SBATCH --job-name=cfg_5.0
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/cfg_5.0_%j.out
#SBATCH --error=logs/cfg_5.0_%j.err

source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --cfg_guidance_scale 5.0 \
         --run_name sweep_cfg5.0 2>&1 | tee eval_results/sweep_cfg5.0.txt
