#!/bin/bash
#SBATCH --job-name=T1_extend
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=logs/T1_extend_%j.out
#SBATCH --error=logs/T1_extend_%j.err

# T1: continue P3 training from epoch 779 up to 1500 epochs, testing whether
# more training improves FID further. ep524 hit FID 59.9; ep779 went to 64.5
# (late-stage overfit signal) — but with more data the curve may dip again.
# Uses the same learned-variance architecture as P3.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/production/_common.sh

LATEST_CKPT=/work/nvme/bgyq/sguan/experiments/exp-3-P3_learned_var/checkpoints/checkpoint_epoch_779.pth
if [ ! -f "$LATEST_CKPT" ]; then
    echo "ERROR: checkpoint not found: $LATEST_CKPT"
    exit 1
fi

run_train --run_name T1_extend \
          --unet_ch 256 --batch_size 512 --num_epochs 1500 \
          --variance_type learned_range --prediction_type epsilon --vlb_weight 0.001 \
          --resume "$LATEST_CKPT"
