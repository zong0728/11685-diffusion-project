#!/bin/bash
#SBATCH --job-name=T1_s3_2
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/T1_s3_2_%j.out
#SBATCH --error=logs/T1_s3_2_%j.err

# Inverted schedule (high-to-low) — sanity check. If this performs worse than the
# low-to-high schedules, it validates that low early / high late is the right direction.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range \
         --num_inference_steps 100 --clip_sample False \
         --cfg_schedule_low 3.0 --cfg_schedule_high 2.0 \
         --run_name sweep_T1_cfgsched_3_2 2>&1 | tee eval_results/sweep_T1_cfgsched_3_2.txt
