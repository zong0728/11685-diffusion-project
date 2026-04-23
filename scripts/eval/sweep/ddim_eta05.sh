#!/bin/bash
#SBATCH --job-name=ddim_eta05
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ddim_eta05_%j.out
#SBATCH --error=logs/ddim_eta05_%j.err

# E10b: middle-ground stochasticity (eta=0.5) interpolating DDIM and DDPM.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --cfg_guidance_scale 3.0 \
         --ddim_eta 0.5 \
         --run_name sweep_ddim_eta05 2>&1 | tee eval_results/sweep_ddim_eta05.txt
