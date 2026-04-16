#!/bin/bash
#SBATCH --job-name=A_baseline
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/A_baseline_%j.out
#SBATCH --error=logs/A_baseline_%j.err

# Run A: reference config that B–E compare against.
source /projects/bgyq/sguan/11685-diffusion-project/scripts/overnight/_common.sh
run_train --run_name A_baseline
