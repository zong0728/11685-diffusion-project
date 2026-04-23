#!/bin/bash
#SBATCH --job-name=abl_noema
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/abl_noema_%j.out
#SBATCH --error=logs/abl_noema_%j.err

# E8: ablation — use the raw (non-EMA) weights at epoch 524 to quantify
# how much EMA contributes to FID. load_checkpoint detects 'unet_state_dict'
# in checkpoint_epoch_*.pth and loads it directly (no EMA path).
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/checkpoint_epoch_779.pth \
         --variance_type learned_range \
         --cfg_guidance_scale 3.0 \
         --run_name sweep_no_ema 2>&1 | tee eval_results/sweep_no_ema.txt
