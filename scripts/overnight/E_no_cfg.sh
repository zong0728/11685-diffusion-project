#!/bin/bash
#SBATCH --job-name=E_nocfg
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/E_nocfg_%j.out
#SBATCH --error=logs/E_nocfg_%j.err

# Run E: disable CFG (plain class-conditional, no unconditional dropout).
source /projects/bgyq/sguan/11685-diffusion-project/scripts/overnight/_common.sh
run_train --run_name E_no_cfg --use_cfg False
