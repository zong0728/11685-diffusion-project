import os
import sys
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import logging
from logging import getLogger as get_logger
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed everything
    seed_everything(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # setup model
    logger.info("Creating model")
    # unet
    unet_output_ch = 2 * args.unet_in_ch if args.variance_type == 'learned_range' else args.unet_in_ch
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch, output_ch=unet_output_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")

    # TODO: ddpm shceduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )
    # vae
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(embed_dim=args.unet_ch, n_classes=args.num_classes, cond_drop_rate=0.0)

    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)

    # scheduler
    if args.use_ddim:
        inference_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            num_inference_steps=args.num_inference_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            prediction_type=args.prediction_type,
            clip_sample=args.clip_sample,
            clip_sample_range=args.clip_sample_range,
        ).to(device)
    else:
        inference_scheduler = scheduler

    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)

    # TODO: pipeline
    pipeline = DDPMPipeline(unet, inference_scheduler, vae=vae, class_embedder=class_embedder)


    logger.info("***** Running Infrence *****")

    unet.eval()
    if class_embedder is not None:
        class_embedder.eval()

    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class
    all_images = []
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device,
            )
            # convert PIL images to tensors
            gen_tensors = torch.stack([transforms.ToTensor()(img) for img in gen_images])
            all_images.append(gen_tensors)
    else:
        # generate 5000 images
        batch_size = args.batch_size
        for _ in tqdm(range(0, 5000, batch_size)):
            cur_batch = min(batch_size, 5000 - len(all_images) * batch_size)
            gen_images = pipeline(
                batch_size=cur_batch,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )
            gen_tensors = torch.stack([transforms.ToTensor()(img) for img in gen_images])
            all_images.append(gen_tensors)

    all_images = torch.cat(all_images, dim=0)[:5000]  # ensure exactly 5000
    logger.info(f"Generated {all_images.shape[0]} images")

    # TODO: load validation images as reference batch
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    val_dir = args.data_dir.replace('train', 'validation')
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)

    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    # compute FID
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # add real images
    for real_images, _ in tqdm(val_loader, desc="Processing real images"):
        fid.update(real_images.to(device), real=True)

    # add generated images
    gen_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(all_images), batch_size=64, shuffle=False
    )
    for (gen_batch,) in tqdm(gen_loader, desc="Processing generated images"):
        fid.update(gen_batch.to(device), real=False)

    fid_score = fid.compute().item()
    logger.info(f"FID Score: {fid_score}")

    # compute IS
    inception_score = InceptionScore(normalize=True).to(device)
    for (gen_batch,) in torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(all_images), batch_size=64, shuffle=False
    ):
        inception_score.update(gen_batch.to(device))

    is_mean, is_std = inception_score.compute()
    logger.info(f"Inception Score: {is_mean.item()} ± {is_std.item()}")

    # optionally generate submission
    try:
        from generate_submission import generate_submission_from_tensors
        # convert to [-1, 1] range for submission
        submission_images = all_images * 2 - 1
        generate_submission_from_tensors(submission_images, output_csv="submission.csv")
        logger.info("Submission CSV saved to submission.csv")
    except Exception as e:
        logger.warning(f"Could not generate submission: {e}")


if __name__ == '__main__':
    main()
