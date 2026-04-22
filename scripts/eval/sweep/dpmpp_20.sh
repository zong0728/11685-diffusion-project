#!/bin/bash
#SBATCH --job-name=dpmpp_20
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dpmpp_20_%j.out
#SBATCH --error=logs/dpmpp_20_%j.err

# S4: DPM-Solver++ (2nd-order, Karras sigmas) at 20 steps on P3 ep524.
# Hypothesis: DPM++ 20 steps rivals DDIM 250 steps in quality, so much cheaper.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --use_ddim False --use_dpmpp True \
         --dpmpp_solver_order 2 --dpmpp_karras True \
         --num_inference_steps 20 \
         --run_name sweep_dpmpp20 2>&1 | tee eval_results/sweep_dpmpp20.txt
