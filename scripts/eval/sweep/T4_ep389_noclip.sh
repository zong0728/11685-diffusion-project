#!/bin/bash
#SBATCH --job-name=T4_ep389
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/T4_ep389_%j.out
#SBATCH --error=logs/T4_ep389_%j.err

# T4 = linear noise schedule. Eval with linear at sampling time.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=/work/nvme/bgyq/sguan/experiments/exp-8-T4_linear/checkpoints/ema_epoch_389.pth
run_eval --ckpt "$CKPT" \
         --variance_type learned_range --beta_schedule linear \
         --num_inference_steps 20 --cfg_guidance_scale 3.0 --clip_sample False \
         --run_name sweep_T4_noclip 2>&1 | tee eval_results/sweep_T4_noclip.txt
