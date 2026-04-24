#!/bin/bash
#SBATCH --job-name=d20_cfg25
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/d20_cfg25_%j.out
#SBATCH --error=logs/d20_cfg25_%j.err

# S5: ddim20 + CFG sweep. Previous CFG sweep used 50 steps; ddim20 might shift the optimum.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --num_inference_steps 20 --cfg_guidance_scale 2.5 \
         --run_name sweep_ddim20_cfg2.5 2>&1 | tee eval_results/sweep_ddim20_cfg2.5.txt
