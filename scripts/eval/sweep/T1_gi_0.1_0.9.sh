#!/bin/bash
#SBATCH --job-name=T1_gi1090
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/T1_gi1090_%j.out
#SBATCH --error=logs/T1_gi1090_%j.err

# Wider interval: apply CFG in the middle 80% of steps, skip only extremes.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range \
         --num_inference_steps 100 --cfg_guidance_scale 3.0 --clip_sample False \
         --guidance_interval_start 0.1 --guidance_interval_end 0.9 \
         --run_name sweep_T1_gi_0.1_0.9 2>&1 | tee eval_results/sweep_T1_gi_0.1_0.9.txt
