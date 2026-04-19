#!/bin/bash
#SBATCH --job-name=P3_resume
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=47:30:00
#SBATCH --output=logs/P3_resume_%j.out
#SBATCH --error=logs/P3_resume_%j.err

# Auto-resume wrapper for P3. Submit with:
#   sbatch --dependency=afterany:<P3_jobid> P3_resume.sh
#
# Finds the latest checkpoint_epoch_*.pth in P3's run directory and resumes training.
# If P3 ran clean to 780/780 → the resume job still fires (afterany), sees start_epoch >= num_epochs,
# and exits without training. That's a harmless no-op, not wasted compute.
# If P3 died mid-run (TIMEOUT / NODE_FAIL / I/O error) → we pick up from the latest
# save_every=25 checkpoint (so we lose at most 24 epochs of progress).

source /projects/bgyq/sguan/11685-diffusion-project/scripts/production/_common.sh

RUN_DIR=$(ls -dt /work/nvme/bgyq/sguan/experiments/exp-*-P3_learned_var 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: no P3_learned_var run directory found under /work/nvme/bgyq/sguan/experiments/"
    echo "Either P3 never started, or it wrote to a different name. Aborting resume."
    exit 1
fi

LATEST_CKPT=$(ls "$RUN_DIR"/checkpoints/checkpoint_epoch_*.pth 2>/dev/null \
    | awk -F'checkpoint_epoch_|\\.pth' '{print $2" "$0}' \
    | sort -n -k1 \
    | tail -1 \
    | cut -d' ' -f2-)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: no checkpoints in $RUN_DIR/checkpoints/. P3 may have crashed before first save."
    echo "Falling back to fresh P3 run (no --resume)."
    run_train --run_name P3_learned_var_resume_fresh \
              --unet_ch 256 --batch_size 512 --num_epochs 780 \
              --variance_type learned_range --prediction_type epsilon --vlb_weight 0.001
    exit $?
fi

echo "Resuming from $LATEST_CKPT"
# Reuse the original run directory so ckpts continue to accumulate in one place.
# Passing --output_dir=<parent>/<run_name> style is awkward; instead we set --run_name to
# the exact existing directory basename so train.py finds it again.
RUN_NAME=$(basename "$RUN_DIR")
# train.py auto-increments exp-N prefixes when run_name exists as a directory; strip the prefix
# so the fresh run_name matches what args.run_name was originally. The resume itself doesn't
# need to write into the old dir — ckpts in the new dir are fine, we just want the weights.
STRIPPED_NAME=$(echo "$RUN_NAME" | sed -E 's/^exp-[0-9]+-//')

run_train --run_name "${STRIPPED_NAME}_resume" \
          --unet_ch 256 --batch_size 512 --num_epochs 780 \
          --variance_type learned_range --prediction_type epsilon --vlb_weight 0.001 \
          --resume "$LATEST_CKPT"
