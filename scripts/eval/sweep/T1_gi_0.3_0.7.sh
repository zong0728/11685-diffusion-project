#!/bin/bash
#SBATCH --job-name=T1_gi3070
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/T1_gi3070_%j.out
#SBATCH --error=logs/T1_gi3070_%j.err

# Narrower interval (middle 40%). More literal Kynkäänniemi 2024 — guidance
# only applies in a small mid-range, rest is completely unguided.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range \
         --num_inference_steps 100 --cfg_guidance_scale 3.0 --clip_sample False \
         --guidance_interval_start 0.3 --guidance_interval_end 0.7 \
         --run_name sweep_T1_gi_0.3_0.7 2>&1 | tee eval_results/sweep_T1_gi_0.3_0.7.txt
