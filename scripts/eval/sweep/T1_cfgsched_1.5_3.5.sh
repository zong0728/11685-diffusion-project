#!/bin/bash
#SBATCH --job-name=T1_s15_35
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/T1_s15_35_%j.out
#SBATCH --error=logs/T1_s15_35_%j.err

# Milder schedule: 1.5 → 3.5. Narrower range, stays closer to proven cfg=2.8 mean.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range \
         --num_inference_steps 100 --clip_sample False \
         --cfg_schedule_low 1.5 --cfg_schedule_high 3.5 \
         --run_name sweep_T1_cfgsched_1.5_3.5 2>&1 | tee eval_results/sweep_T1_cfgsched_1.5_3.5.txt
