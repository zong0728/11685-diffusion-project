#!/bin/bash
#SBATCH --job-name=T1_dpmpp30
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/T1_dpmpp30_%j.out
#SBATCH --error=logs/T1_dpmpp30_%j.err

# DPM++ retry — fixed by exposing N-1 timesteps to the pipeline so diffusers'
# look-ahead never indexes past the sigmas tensor. Uses Karras sigmas internally
# (diffusers' use_karras_sigmas=True), which is independent of our own Karras bug.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" \
         --variance_type learned_range \
         --use_ddim False --use_dpmpp True \
         --dpmpp_solver_order 2 --dpmpp_karras True \
         --num_inference_steps 30 --cfg_guidance_scale 3.0 --clip_sample False \
         --run_name sweep_T1_dpmpp30_v2 2>&1 | tee eval_results/sweep_T1_dpmpp30_v2.txt
