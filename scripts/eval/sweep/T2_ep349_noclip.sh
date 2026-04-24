#!/bin/bash
#SBATCH --job-name=T2_ep349
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/T2_ep349_%j.out
#SBATCH --error=logs/T2_ep349_%j.err

# T2 = ch=320, v-prediction (no learned variance — old config). Eval with vpred + cosine.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/eval/_common.sh
CKPT=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T2_ch384/checkpoints/ema_epoch_*.pth 2>/dev/null | head -1)
run_eval --ckpt "$CKPT" \
         --variance_type fixed_small --prediction_type v_prediction \
         --unet_ch 320 \
         --num_inference_steps 20 --cfg_guidance_scale 3.0 --clip_sample False \
         --run_name sweep_T2_noclip 2>&1 | tee eval_results/sweep_T2_noclip.txt
