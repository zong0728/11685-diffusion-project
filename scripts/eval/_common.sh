# Shared env + defaults for FID/IS evaluation of overnight ablation ckpts.
# Each caller sources this, sets --ckpt + any architecture overrides, and
# invokes `run_eval "$@"`.

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
export PYTHONUSERBASE=$HOME/.local
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

cd /projects/bgyq/sguan/11685-diffusion-project
mkdir -p logs eval_results
export WANDB_MODE=offline

# Architecture defaults match overnight A/B/D/E (ch=256). C overrides ch=320.
COMMON_ARGS=(
    --config configs/ddpm.yaml
    --data_dir /work/nvme/bgyq/sguan/imagenet100_128x128/train
    --image_size 128 --unet_in_size 32 --unet_in_ch 3
    --unet_ch 256 --unet_ch_mult 1 2 3 4 --unet_attn 2 3 --unet_num_res_blocks 2 --unet_dropout 0.0
    --num_classes 100 --num_workers 8
    --num_train_timesteps 1000 --num_inference_steps 50 --beta_schedule cosine
    --latent_ddpm True --use_ddim True
    --use_cfg True --cfg_guidance_scale 2.0
    --mixed_precision bf16
    --prediction_type epsilon
    --batch_size 50
)

run_eval() {
    python inference.py "${COMMON_ARGS[@]}" "$@"
}
