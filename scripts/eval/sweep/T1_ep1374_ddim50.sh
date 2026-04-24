#!/bin/bash
#SBATCH --job-name=T1_e1374
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/T1_e1374_%j.out
#SBATCH --error=logs/T1_e1374_%j.err

# Earlier T1 ckpt at our winning ddim50 setting. Round 4 found ep1499 ≈ 1474 ≈ 1449
# all tied at ~26.5 with ddim20, but maybe the curve looks different at ddim50.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume/checkpoints/ema_epoch_1374.pth | head -1)
run_eval --ckpt "$CKPT" --variance_type learned_range \
         --num_inference_steps 50 --cfg_guidance_scale 3.0 --clip_sample False \
         --run_name sweep_T1_ep1374_ddim50 2>&1 | tee eval_results/sweep_T1_ep1374_ddim50.txt
