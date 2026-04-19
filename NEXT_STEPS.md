# Morning-of status — 2026-04-19

## What happened overnight

I implemented **Improved-DDPM-style learned variance** (Nichol & Dhariwal 2021)
as the next advanced technique, per TA's PPT suggestions.

Originally the plan was "A (swap VAE to 4-channel) + B (learned variance)".
I had to **drop A** — `models/vae.py` has an explicit
"do not change anything in __init__" constraint that makes swapping to a 4-channel
SD VAE a direct rule violation. So this round is learned variance only.

## Code changes (commit summary)

- `models/unet.py` — added `output_ch` arg (default=input_ch for backward compat).
  Under `variance_type=learned_range` the head emits `2*C` channels: first C = eps,
  last C = per-pixel variance interpolation coefficients.
- `schedulers/scheduling_ddpm.py`
  - `_split_model_output` helper to strip variance channels in the step() path.
  - New `_get_variance` branch `learned_range` that blends log(beta_tilde_t) and log(beta_t)
    by `frac = (var_coef+1)/2`.
  - `vlb_terms()` closed-form KL between posterior q(x_{t-1}|x_t,x_0) and model p_theta,
    with stop-grad on eps_pred so VLB only trains the variance head.
- `schedulers/scheduling_ddim.py` — discards variance channels (DDIM is deterministic-ish).
- `pipelines/ddpm.py` — CFG now applied only to the eps half; variance coefs pass through
  from the conditional branch (Improved DDPM appendix B convention).
- `train.py`
  - CLI: `--variance_type learned_range` and `--vlb_weight 0.001`.
  - UNet output_ch auto-set to 2*input_ch when learned_range.
  - Hybrid loss: `L_simple + λ·L_vlb`, λ=0.001.
- `inference.py` — mirrors train.py for correct UNet construction.
- `scripts/production/smoke_learned_var.sh` — 1h, 2-epoch smoke test.
- `scripts/production/P3_learned_var.sh` — 47.5h, 780-epoch production run.

## What the morning person (you) should do

**Step 1 — pull latest code:**
```
cd /projects/bgyq/sguan/11685-diffusion-project
git pull
```

**Step 2 — run the smoke test first.** DO NOT launch P3 directly.
```
cd scripts/production
sbatch smoke_learned_var.sh
squeue -u $USER
```
Wait for it to finish (1h budget, but should actually take ~10-15 min for 2 epochs).
Check the `.err` file for loss values that look sane (should decrease from epoch 0 to 1).

**Step 3 — if smoke passed, launch P3:**
```
sbatch P3_learned_var.sh
squeue -u $USER
```
Finishes ~2026-04-21 early morning.

**Step 4 — if smoke FAILED:** don't panic. The 5 overnight ckpts + P1/P2 partial ckpts
are untouched. Worst case we write the report with what we have (already has real FID numbers).

## Running / existing stuff to NOT touch

- `experiments/exp-11/12/13/14/15-*` — the 5 overnight ablation runs, 3 EMA ckpts each
- `/work/nvme/bgyq/sguan/experiments/exp-0-P1_ch256` and `exp-1-P2_ch320` — partial
  production ckpts (epochs 324 and 249). Crashed mid-run due to I/O error + NODE_FAIL,
  not our fault. Still FID-evaluable as a backup.

## What still needs doing after P3 finishes

1. FID-evaluate P3's final EMA ckpt (also useful: evaluate P1/P2 at epoch 249/324 for
   direct learned-variance-vs-fixed-variance comparison → ablation point for report)
2. CFG scale sweep (`w` in {1, 2, 4, 7.5}) on the winning ckpt — inference-time only
3. DDIM steps sweep (50, 100, 250) on the winning ckpt
4. Pick winner → Kaggle submission CSV via `generate_submission.py`
5. Record video (4/24 deadline)
6. Write final report (4/29 deadline)
