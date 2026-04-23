#!/bin/bash
#SBATCH --job-name=T1_resume
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/T1_resume_%j.out
#SBATCH --error=logs/T1_resume_%j.err

# T1 chained resume. Submit with:
#   sbatch --dependency=afterany:<T1_jobid> T1_resume.sh
# Picks up the latest checkpoint_epoch_*.pth from the T1_extend run dir and
# continues training to epoch 1500. If T1 already finished cleanly, this
# resume sees start_epoch >= num_epochs and exits as a no-op.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/production/_common.sh

RUN_DIR=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-T1_extend 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: no T1_extend run directory found"
    exit 1
fi

LATEST_CKPT=$(ls "$RUN_DIR"/checkpoints/checkpoint_epoch_*.pth 2>/dev/null \
    | awk -F'checkpoint_epoch_|\\.pth' '{print $2" "$0}' \
    | sort -n -k1 | tail -1 | cut -d' ' -f2-)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: no checkpoint in $RUN_DIR/checkpoints/"
    exit 1
fi

echo "Resuming T1 from $LATEST_CKPT"
run_train --run_name T1_extend_resume \
          --unet_ch 256 --batch_size 512 --num_epochs 1500 \
          --variance_type learned_range --prediction_type epsilon --vlb_weight 0.001 \
          --resume "$LATEST_CKPT"
