#!/bin/bash
#SBATCH --job-name=ddim_eta1
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ddim_eta1_%j.out
#SBATCH --error=logs/ddim_eta1_%j.err

# E10: stochastic DDIM (eta=1.0, equivalent to DDPM ancestral sampling).
# Hypothesis: stochasticity helps with mode coverage, possibly lower FID.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --cfg_guidance_scale 3.0 \
         --ddim_eta 1.0 \
         --run_name sweep_ddim_eta1 2>&1 | tee eval_results/sweep_ddim_eta1.txt
