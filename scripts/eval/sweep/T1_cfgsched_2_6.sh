#!/bin/bash
#SBATCH --job-name=T1_s2_6
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/T1_s2_6_%j.out
#SBATCH --error=logs/T1_s2_6_%j.err

# 2.0 → 5.0 was our winner (FID 22.16). Push the high end further.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range \
         --num_inference_steps 100 --clip_sample False \
         --cfg_schedule_low 2.0 --cfg_schedule_high 6.0 \
         --run_name sweep_T1_cfgsched_2_6 2>&1 | tee eval_results/sweep_T1_cfgsched_2_6.txt
