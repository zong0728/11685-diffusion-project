#!/bin/bash
#SBATCH --job-name=eval_C
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/eval_C_%j.out
#SBATCH --error=logs/eval_C_%j.err

# FID/IS for C_bigmodel (ch=320). Architecture override.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt experiments/exp-13-C_bigmodel/checkpoints/ema_epoch_108.pth \
         --unet_ch 320 \
         --run_name eval_C_bigmodel 2>&1 | tee eval_results/C_bigmodel.txt
