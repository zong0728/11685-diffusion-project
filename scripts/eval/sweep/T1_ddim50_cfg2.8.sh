#!/bin/bash
#SBATCH --job-name=T1_d50c28
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/T1_d50c28_%j.out
#SBATCH --error=logs/T1_d50c28_%j.err

source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range \
         --num_inference_steps 50 --cfg_guidance_scale 2.8 --clip_sample False \
         --run_name sweep_T1_ddim50_cfg2.8 2>&1 | tee eval_results/sweep_T1_ddim50_cfg2.8.txt
