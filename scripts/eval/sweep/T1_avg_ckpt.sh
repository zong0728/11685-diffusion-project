#!/bin/bash
#SBATCH --job-name=T1_avg
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/T1_avg_%j.out
#SBATCH --error=logs/T1_avg_%j.err

# Average the last 3 T1 EMA ckpts (ep1449, ep1474, ep1499) and eval.
# Checkpoint averaging (SWA-style) typically lowers FID by 1-3 points
# when the model is in a stable region of weight space.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh

T1_DIR=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend_resume | head -1)
AVG=/work/nvme/bgyq/sguan/experiments/T1_avg_3ckpt.pth

echo "Building averaged ckpt from T1 ep1449 + ep1474 + ep1499..."
python /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/sweep/make_avg_ckpt.py \
    --ckpts \
        "$T1_DIR/checkpoints/ema_epoch_1449.pth" \
        "$T1_DIR/checkpoints/ema_epoch_1474.pth" \
        "$T1_DIR/checkpoints/ema_epoch_1499.pth" \
    --out "$AVG"

run_eval --ckpt "$AVG" --variance_type learned_range \
         --num_inference_steps 25 --cfg_guidance_scale 3.0 --clip_sample False \
         --run_name sweep_T1_avg3 2>&1 | tee eval_results/sweep_T1_avg3.txt
