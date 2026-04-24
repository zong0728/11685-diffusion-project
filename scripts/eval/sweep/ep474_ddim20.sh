#!/bin/bash
#SBATCH --job-name=ep474_d20
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ep474_d20_%j.out
#SBATCH --error=logs/ep474_d20_%j.err

# S2c: ep474 — between ep449 and the winner ep524.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_474.pth \
         --variance_type learned_range \
         --num_inference_steps 20 --cfg_guidance_scale 3.0 \
         --run_name sweep_ep474_ddim20 2>&1 | tee eval_results/sweep_ep474_ddim20.txt
