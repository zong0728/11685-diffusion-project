#!/bin/bash
#SBATCH --job-name=noclip_40
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/noclip_40_%j.out
#SBATCH --error=logs/noclip_40_%j.err

source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --num_inference_steps 20 --cfg_guidance_scale 4.0 --clip_sample False \
         --run_name sweep_noclip_cfg4.0 2>&1 | tee eval_results/sweep_noclip_cfg4.0.txt
