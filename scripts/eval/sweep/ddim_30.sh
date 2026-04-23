#!/bin/bash
#SBATCH --job-name=ddim_30
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ddim_30_%j.out
#SBATCH --error=logs/ddim_30_%j.err

# E2: DDIM 30 steps + cfg=3.0.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --num_inference_steps 30 --cfg_guidance_scale 3.0 \
         --run_name sweep_ddim30_cfg3 2>&1 | tee eval_results/sweep_ddim30_cfg3.txt
