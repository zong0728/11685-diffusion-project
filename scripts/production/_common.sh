# Shared env + defaults for 48h production training.
# Differences from scripts/overnight/_common.sh:
#   - Output goes to /work/nvme (NVMe, 8.5 PB, vs /projects Lustre quota)
#   - --save_every 25 to avoid the per-epoch ckpt flood that burned /projects
#   - num_epochs scaled to fill ~47h (leave buffer before TIME LIMIT kill)
# v-prediction + CFG + min-SNR kept — FID ablation picked these as the winners.

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
export PYTHONUSERBASE=$HOME/.local
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

cd /projects/bgyq/sguan/11685-diffusion-project
mkdir -p logs /work/nvme/bgyq/sguan/experiments
export WANDB_MODE=offline

COMMON_ARGS=(
    --config configs/ddpm.yaml
    --data_dir /work/nvme/bgyq/sguan/imagenet100_128x128/train
    --output_dir /work/nvme/bgyq/sguan/experiments
    --image_size 128 --unet_in_size 32 --unet_in_ch 3
    --unet_ch_mult 1 2 3 4 --unet_attn 2 3 --unet_num_res_blocks 2 --unet_dropout 0.0
    --num_workers 16 --num_classes 100
    --learning_rate 1e-4 --weight_decay 1e-4
    --num_train_timesteps 1000 --num_inference_steps 50 --beta_schedule cosine
    --latent_ddpm True --use_ddim True
    --use_cfg True --cfg_guidance_scale 2.0
    --use_ema True --ema_decay 0.9999
    --mixed_precision bf16 --min_snr_gamma 5.0
    --prediction_type v_prediction
    --save_every 25
)

run_train() {
    python train.py "${COMMON_ARGS[@]}" "$@"
}
