#!/bin/bash
#SBATCH --job-name=T5_cfg25
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/T5_cfg25_%j.out
#SBATCH --error=logs/T5_cfg25_%j.err

# T5 (sigmoid) hit 31.37 at cfg=3 — sigmoid schedule literature says it prefers different CFG.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T5_sigmoid/checkpoints/ema_epoch_*.pth 2>/dev/null | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range --beta_schedule sigmoid \
         --num_inference_steps 20 --cfg_guidance_scale 2.5 --clip_sample False \
         --run_name sweep_T5_cfg2.5 2>&1 | tee eval_results/sweep_T5_cfg2.5.txt
