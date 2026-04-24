#!/bin/bash
#SBATCH --job-name=ddim15_cfg3
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ddim15_cfg3_%j.out
#SBATCH --error=logs/ddim15_cfg3_%j.err

# S1b: 15 DDIM steps — between 10 and 20 to nail the sweet spot.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --num_inference_steps 15 --cfg_guidance_scale 3.0 \
         --run_name sweep_ddim15_cfg3 2>&1 | tee eval_results/sweep_ddim15_cfg3.txt
