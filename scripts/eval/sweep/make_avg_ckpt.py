"""
Average several EMA checkpoints into one. The standard "checkpoint averaging"
trick — Izmailov 2018 (SWA) and later many diffusion papers showed that simply
averaging the last few EMA snapshots of a stable model can lower FID by 1-3.

Reads N ckpts, averages their ema_state_dict tensors, writes a new ckpt that
load_checkpoint() can consume just like a regular ema ckpt.
"""
import argparse
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', required=True, help='paths to ema_epoch_*.pth files to average')
    p.add_argument('--out', required=True, help='output path for averaged ckpt')
    args = p.parse_args()

    print(f"Loading {len(args.ckpts)} checkpoints...")
    state_dicts = []
    last_epoch = 0
    for path in args.ckpts:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if 'ema_state_dict' not in ck:
            print(f"ERROR: {path} has no 'ema_state_dict' key. Found: {list(ck.keys())}", file=sys.stderr)
            sys.exit(1)
        state_dicts.append(ck['ema_state_dict'])
        last_epoch = max(last_epoch, ck.get('epoch', 0))
        print(f"  loaded {path} (epoch {ck.get('epoch', '?')})")

    # Average tensors element-wise. All ckpts must have identical keys + shapes.
    avg = {}
    keys = list(state_dicts[0].keys())
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        avg[k] = stacked.mean(dim=0)
    print(f"Averaged {len(keys)} tensors across {len(state_dicts)} ckpts.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'ema_state_dict': avg, 'epoch': last_epoch}, out_path)
    print(f"Wrote averaged ckpt to {out_path}")


if __name__ == '__main__':
    main()
