#!/bin/bash
#SBATCH --job-name=T1_s14_v2
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/T1_s14_v2_%j.out
#SBATCH --error=logs/T1_s14_v2_%j.err

# Retry cfgsched 1.0 -> 4.0 after fixing the None-class-emb crash.
# Original cfgsched_1_4 failed in 19s because cfg=1.0 at step 0 took the
# non-CFG pipeline branch (c=None), then UNet cross-attn LayerNorm'd None.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range \
         --num_inference_steps 100 --clip_sample False \
         --cfg_schedule_low 1.0 --cfg_schedule_high 4.0 \
         --run_name sweep_T1_cfgsched_1_4_v2 2>&1 | tee eval_results/sweep_T1_cfgsched_1_4_v2.txt
