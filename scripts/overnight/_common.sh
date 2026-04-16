# Sourced by A/B/C/D/E overnight scripts — shared env + defaults.
# Each caller sources this, then sets any run-specific flags and invokes
# `run_train "$@"` (or calls python directly with overrides).

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
export PYTHONUSERBASE=$HOME/.local
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

cd /projects/bgyq/sguan/11685-diffusion-project
mkdir -p logs experiments
export WANDB_MODE=offline

# Defaults shared by every overnight ablation. Caller overrides any of these
# by passing the corresponding --flag after "$@" when invoking run_train.
COMMON_ARGS=(
    --config configs/ddpm.yaml
    --data_dir /work/nvme/bgyq/sguan/imagenet100_128x128/train
    --image_size 128 --unet_in_size 32 --unet_in_ch 3
    --unet_ch 256 --unet_ch_mult 1 2 3 4 --unet_attn 2 3 --unet_num_res_blocks 2 --unet_dropout 0.0
    --num_epochs 150 --batch_size 512 --num_workers 16 --num_classes 100
    --learning_rate 1e-4 --weight_decay 1e-4
    --num_train_timesteps 1000 --num_inference_steps 50 --beta_schedule cosine
    --latent_ddpm True --use_ddim True
    --use_cfg True --cfg_guidance_scale 2.0
    --use_ema True --ema_decay 0.9999
    --mixed_precision bf16 --min_snr_gamma 5.0
    --prediction_type epsilon
)

run_train() {
    python train.py "${COMMON_ARGS[@]}" "$@"
}
