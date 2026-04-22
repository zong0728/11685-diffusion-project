#!/bin/bash
#SBATCH --job-name=ddim_100
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ddim_100_%j.out
#SBATCH --error=logs/ddim_100_%j.err

# S2: DDIM steps sweep. ep524, cfg=2.0 (default), steps=100 (vs our default 50).
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --num_inference_steps 100 \
         --run_name sweep_ddim100 2>&1 | tee eval_results/sweep_ddim100.txt
