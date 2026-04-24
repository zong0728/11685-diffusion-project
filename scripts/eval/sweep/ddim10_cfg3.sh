#!/bin/bash
#SBATCH --job-name=ddim10_cfg3
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ddim10_cfg3_%j.out
#SBATCH --error=logs/ddim10_cfg3_%j.err

# S1: fewest DDIM steps — 250→100→50→30→20 was monotonically improving. Test 10.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --num_inference_steps 10 --cfg_guidance_scale 3.0 \
         --run_name sweep_ddim10_cfg3 2>&1 | tee eval_results/sweep_ddim10_cfg3.txt
