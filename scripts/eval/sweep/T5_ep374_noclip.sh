#!/bin/bash
#SBATCH --job-name=T5_ep374
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/T5_ep374_%j.out
#SBATCH --error=logs/T5_ep374_%j.err

# T5 = sigmoid schedule. Eval with sigmoid at sampling time.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T5_sigmoid/checkpoints/ema_epoch_*.pth 2>/dev/null | head -1)
run_eval --ckpt "$CKPT" \
         --variance_type learned_range --beta_schedule sigmoid \
         --num_inference_steps 20 --cfg_guidance_scale 3.0 --clip_sample False \
         --run_name sweep_T5_noclip 2>&1 | tee eval_results/sweep_T5_noclip.txt
