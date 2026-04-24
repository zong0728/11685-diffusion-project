#!/bin/bash
#SBATCH --job-name=abl_noclip
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/abl_noclip_%j.out
#SBATCH --error=logs/abl_noclip_%j.err

# S4: clip_sample=False ablation. Default clip_sample=True bounds the predicted x_0
# to [-1, 1] each step, which can suppress diversity in latent space (where the
# distribution isn't actually bounded by the VAE).
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range \
         --num_inference_steps 20 --cfg_guidance_scale 3.0 \
         --clip_sample False \
         --run_name sweep_no_clip 2>&1 | tee eval_results/sweep_no_clip.txt
