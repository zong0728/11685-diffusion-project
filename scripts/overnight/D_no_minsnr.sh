#!/bin/bash
#SBATCH --job-name=D_nominsnr
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/D_nominsnr_%j.out
#SBATCH --error=logs/D_nominsnr_%j.err

# Run D: disable min-SNR weighting (gamma=0 → uniform loss).
source /projects/bgyq/sguan/11685-diffusion-project/scripts/overnight/_common.sh
run_train --run_name D_no_minsnr --min_snr_gamma 0.0
