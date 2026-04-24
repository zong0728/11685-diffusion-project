#!/bin/bash
#SBATCH --job-name=T1_dpmpp25
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/T1_dpmpp25_%j.out
#SBATCH --error=logs/T1_dpmpp25_%j.err

# DPM-Solver++ on T1 ep1499 — fixed the IndexError last round, but P3 was already
# noclip-saturated when we tested. Now T1 is the better base, retry DPM++ here.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" \
         --variance_type learned_range \
         --use_ddim False --use_dpmpp True \
         --dpmpp_solver_order 2 --dpmpp_karras True \
         --num_inference_steps 25 --cfg_guidance_scale 3.0 --clip_sample False \
         --run_name sweep_T1_dpmpp25 2>&1 | tee eval_results/sweep_T1_dpmpp25.txt
