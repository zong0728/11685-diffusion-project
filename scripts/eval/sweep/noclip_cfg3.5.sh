#!/bin/bash
#SBATCH --job-name=noclip_35
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/noclip_35_%j.out
#SBATCH --error=logs/noclip_35_%j.err

# Bracket the no_clip CFG sweet spot. cfg=3.0 → 28.13, cfg=2.5 → 28.15. Try 3.5.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --num_inference_steps 20 --cfg_guidance_scale 3.5 --clip_sample False \
         --run_name sweep_noclip_cfg3.5 2>&1 | tee eval_results/sweep_noclip_cfg3.5.txt
