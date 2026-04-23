#!/bin/bash
#SBATCH --job-name=abl_fixvar
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/abl_fixvar_%j.out
#SBATCH --error=logs/abl_fixvar_%j.err

# E4: ablation — use the same P3 ckpt but pretend variance is fixed_small
# instead of learned. The DDIMScheduler will ignore var_coef channels
# (they're stripped in step()). Tells us if the learned-variance
# coefficients are pulling weight at sampling time, or just during training.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
run_eval --ckpt /work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/ema_epoch_524.pth \
         --variance_type learned_range --sampling_variance_type fixed_small \
         --cfg_guidance_scale 3.0 \
         --run_name sweep_fixed_var 2>&1 | tee eval_results/sweep_fixed_var.txt
