#!/bin/bash
#SBATCH --job-name=ep779_ema
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ep779_ema_%j.out
#SBATCH --error=logs/ep779_ema_%j.err

# Companion to ablation_no_ema.sh: EMA weights at the same epoch (779) to form
# a clean EMA vs no-EMA pair. Uses cfg=3.0 so the number lands on our sweep table.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_779.pth \
         --variance_type learned_range \
         --cfg_guidance_scale 3.0 \
         --run_name sweep_ep779_ema 2>&1 | tee eval_results/sweep_ep779_ema.txt
