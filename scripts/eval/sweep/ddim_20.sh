#!/bin/bash
#SBATCH --job-name=ddim_20
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ddim_20_%j.out
#SBATCH --error=logs/ddim_20_%j.err

# E2: fewer DDIM steps — 50→100→250 was monotonically worse, try 20.
# Combine with the winning CFG=3.0 from previous sweep.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --num_inference_steps 20 --cfg_guidance_scale 3.0 \
         --run_name sweep_ddim20_cfg3 2>&1 | tee eval_results/sweep_ddim20_cfg3.txt
