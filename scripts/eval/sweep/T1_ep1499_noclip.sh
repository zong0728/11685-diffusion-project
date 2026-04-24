#!/bin/bash
#SBATCH --job-name=T1_ep1499
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/T1_ep1499_%j.out
#SBATCH --error=logs/T1_ep1499_%j.err

# Eval T1_extend_resume final ckpt @ ep1499 with our new winner config (no_clip + ddim20 + cfg3).
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" \
         --variance_type learned_range \
         --num_inference_steps 20 --cfg_guidance_scale 3.0 --clip_sample False \
         --run_name sweep_T1_ep1499_noclip 2>&1 | tee eval_results/sweep_T1_ep1499_noclip.txt
