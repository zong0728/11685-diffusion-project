#!/bin/bash
#SBATCH --job-name=T1_gisch
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/T1_gisch_%j.out
#SBATCH --error=logs/T1_gisch_%j.err

# Combine our two wins: CFG schedule 2→5 + guidance interval 0.2-0.8.
# Within the interval, schedule interpolates 2→5 (but only on the active 60% of steps).
# Outside the interval, cfg=1.0. See if they stack.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range \
         --num_inference_steps 100 --clip_sample False \
         --cfg_schedule_low 2.0 --cfg_schedule_high 5.0 \
         --guidance_interval_start 0.2 --guidance_interval_end 0.8 \
         --run_name sweep_T1_gi_sched 2>&1 | tee eval_results/sweep_T1_gi_sched.txt
