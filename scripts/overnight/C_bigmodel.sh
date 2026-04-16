#!/bin/bash
#SBATCH --job-name=C_bigmodel
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/C_bigmodel_%j.out
#SBATCH --error=logs/C_bigmodel_%j.err

# Run C: bigger UNet (ch=320, ~280M params). batch=384 to fit in 98GB.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/overnight/_common.sh
run_train --run_name C_bigmodel --unet_ch 320 --batch_size 384 --num_epochs 110
