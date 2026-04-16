#!/bin/bash
#SBATCH --job-name=B_vpred
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/B_vpred_%j.out
#SBATCH --error=logs/B_vpred_%j.err

# Run B: v-prediction target (vs A's epsilon).
source /projects/bgyq/sguan/11685-diffusion-project/scripts/overnight/_common.sh
run_train --run_name B_vpred --prediction_type v_prediction
