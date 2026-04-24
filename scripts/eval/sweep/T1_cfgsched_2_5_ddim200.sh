#!/bin/bash
#SBATCH --job-name=T1_s2_5_d200
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/T1_s2_5_d200_%j.out
#SBATCH --error=logs/T1_s2_5_d200_%j.err

# Winner CFG schedule + 200 DDIM steps (ddim200 alone was 23.36, pair with the
# new CFG schedule for compound gain).
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1499.pth | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range \
         --num_inference_steps 200 --clip_sample False \
         --cfg_schedule_low 2.0 --cfg_schedule_high 5.0 \
         --run_name sweep_T1_cfgsched_2_5_ddim200 2>&1 | tee eval_results/sweep_T1_cfgsched_2_5_ddim200.txt
