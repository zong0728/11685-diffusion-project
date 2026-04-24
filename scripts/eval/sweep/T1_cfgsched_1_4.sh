#!/bin/bash
#SBATCH --job-name=T1_s1_4
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/T1_s1_4_%j.out
#SBATCH --error=logs/T1_s1_4_%j.err

# CFG schedule: start at 1.0 (no guidance, early/high-noise steps), ramp to 4.0
# (strong guidance, late/low-noise steps). Chen 2023 suggests low CFG at high
# noise is better for diversity — saves FID vs constant-high CFG.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range \
         --num_inference_steps 100 --clip_sample False \
         --cfg_schedule_low 1.0 --cfg_schedule_high 4.0 \
         --run_name sweep_T1_cfgsched_1_4 2>&1 | tee eval_results/sweep_T1_cfgsched_1_4.txt
