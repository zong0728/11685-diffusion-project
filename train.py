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
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint, EMA


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    
    # config file
    parser.add_argument("--config", type=str, default='configs/ddpm.yaml', help="config file used to specify parameters")

    # data 
    parser.add_argument("--data_dir", type=str, default='./data/imagenet100_128x128/train', help="data folder") 
    parser.add_argument("--image_size", type=int, default=128, help="image size")
    parser.add_argument("--batch_size", type=int, default=4, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    parser.add_argument("--num_classes", type=int, default=100, help="number of classes in dataset")

    # training
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1, help="save checkpoint every N epochs (last epoch always saved)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--mixed_precision", type=str, default='none', choices=['fp16', 'bf16', 'fp32', 'none'], help='mixed precision')
    
    # ddpm
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="ddpm training timesteps")
    parser.add_argument("--num_inference_steps", type=int, default=200, help="ddpm inference timesteps")
    parser.add_argument("--beta_start", type=float, default=0.0002, help="ddpm beta start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="ddpm beta end")
    parser.add_argument("--beta_schedule", type=str, default='linear', help="ddpm beta schedule")
    parser.add_argument("--variance_type", type=str, default='fixed_small', help="ddpm variance type (fixed_small / fixed_large / learned_range)")
    parser.add_argument("--vlb_weight", type=float, default=0.001, help="weight on VLB loss term when variance_type=='learned_range' (Nichol 2021 uses 0.001)")
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="ddpm epsilon type")
    parser.add_argument("--clip_sample", type=str2bool, default=True, help="whether to clip sample at each step of reverse process")
    parser.add_argument("--clip_sample_range", type=float, default=1.0, help="clip sample range")
    
    # unet
    parser.add_argument("--unet_in_size", type=int, default=128, help="unet input image size")
    parser.add_argument("--unet_in_ch", type=int, default=3, help="unet input channel size")
    parser.add_argument("--unet_ch", type=int, default=128, help="unet channel size")
    parser.add_argument("--unet_ch_mult", type=int, default=[1, 2, 2, 2], nargs='+', help="unet channel multiplier")
    parser.add_argument("--unet_attn", type=int, default=[1, 2, 3], nargs='+', help="unet attantion stage index")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="unet number of res blocks")
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="unet dropout")
    
    # vae
    parser.add_argument("--latent_ddpm", type=str2bool, default=False, help="use vqvae for latent ddpm")
    
    # cfg
    parser.add_argument("--use_cfg", type=str2bool, default=False, help="use cfg for conditional (latent) ddpm")
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.0, help="cfg for inference")
    
    # ddim sampler for inference
    parser.add_argument("--use_ddim", type=str2bool, default=False, help="use ddim sampler for inference")
    parser.add_argument("--use_dpmpp", type=str2bool, default=False, help="use DPM-Solver++ sampler (overrides use_ddim)")
    parser.add_argument("--dpmpp_solver_order", type=int, default=2, help="DPM-Solver++ order (1/2/3)")
    parser.add_argument("--dpmpp_karras", type=str2bool, default=True, help="use Karras sigmas for DPM-Solver++")

    # EMA (exponential moving average of weights)
    parser.add_argument("--use_ema", type=str2bool, default=True, help="use EMA of UNet weights for inference")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")

    # Loss weighting
    parser.add_argument("--min_snr_gamma", type=float, default=0.0, help="Min-SNR loss weighting gamma (5.0 is a common good value). 0 disables.")

    # torch.compile
    parser.add_argument("--compile_model", type=str2bool, default=False, help="Use torch.compile on the UNet (10-30%% faster on H100/H200).")

    # resume from checkpoint
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume training from")
    
    # checkpoint path for inference
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path for inference")
    
    # first parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args
    
    
def main():
    
    # parse arguments
    args = parse_args()
    
    # seed everything
    seed_everything(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # setup distributed initialize and device
    if not hasattr(args, 'device'):
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    device = init_distributed_device(args)
    if args.distributed:
        logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0
    
    
    # setup dataset
    logger.info("Creating dataset")
    # TODO: use transform to normalize your images to [-1, 1]
    # TODO: you can also use horizontal flip
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # maps [0,1] -> [-1,1]
    ])
    # TOOD: use image folder for your train dataset
    train_dataset = datasets.ImageFolder(args.data_dir, transform=transform)

    # TODO: setup dataloader
    sampler = None
    if args.distributed:
        # TODO: distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # TODO: shuffle
    shuffle = False if sampler else True
    # TODO dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    # calculate total batch_size
    total_batch_size = args.batch_size * args.world_size 
    args.total_batch_size = total_batch_size
    
    # setup experiment folder
    os.makedirs(args.output_dir, exist_ok=True)
    if args.run_name is None:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}'
    else:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}-{args.run_name}'
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, 'checkpoints')
    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
    
    # setup model
    logger.info("Creating model")
    # unet — under learned_range variance the head emits 2*C channels (eps | var_coef).
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
    
    # NOTE: this is for latent DDPM 
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        # NOTE: do not change this
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
        
    # Note: this is for cfg
    class_embedder = None
    if args.use_cfg:
        # TODO:
        class_embedder = ClassEmbedder(embed_dim=args.unet_ch, n_classes=args.num_classes, cond_drop_rate=0.1)
        
    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
    
    # TODO: setup optimizer
    params = list(unet.parameters())
    if class_embedder is not None:
        params += list(class_embedder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    # TODO: setup scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Setup EMA — maintains a slow-moving average of UNet weights for inference
    ema = None
    if getattr(args, 'use_ema', True):
        ema_decay = getattr(args, 'ema_decay', 0.9999)
        ema = EMA(unet, decay=ema_decay)
        ema.ema_model = ema.ema_model.to(device)

    # Setup mixed precision autocast (bf16 is ideal for H100/H200)
    amp_dtype = None
    if args.mixed_precision == 'bf16':
        amp_dtype = torch.bfloat16
    elif args.mixed_precision == 'fp16':
        amp_dtype = torch.float16
    use_amp = amp_dtype is not None
    # bf16 has the same dynamic range as fp32, so no GradScaler is needed.
    # fp16 would need a GradScaler; we don't use fp16 on H200 anyway.

    if use_amp and is_primary(args):
        logger.info(f"Mixed precision enabled: {args.mixed_precision}")

    # Optional torch.compile (PyTorch 2.x graph capture for ~10-30% speedup on transformer-like models)
    if getattr(args, 'compile_model', False):
        if is_primary(args):
            logger.info("Compiling UNet with torch.compile(mode='max-autotune')...")
        unet = torch.compile(unet, mode='max-autotune', fullgraph=False)
        unet_wo_ddp = unet  # compiled module, still usable for sampling

    # Min-SNR loss weighting (Hang et al. 2023, "Efficient Diffusion Training via Min-SNR Weighting Strategy")
    # Gives higher weight to intermediate timesteps; yields ~3x faster convergence in practice.
    min_snr_gamma = getattr(args, 'min_snr_gamma', 0.0)
    if min_snr_gamma > 0 and is_primary(args):
        logger.info(f"Min-SNR loss weighting enabled with gamma={min_snr_gamma}")

    # Resume from checkpoint if requested
    start_epoch = 0
    if getattr(args, 'resume', None):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt['unet_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        # Load EMA if available
        if ema is not None:
            ema_path = args.resume.replace('checkpoint_epoch_', 'ema_epoch_')
            if os.path.exists(ema_path):
                ema_ckpt = torch.load(ema_path, map_location=device, weights_only=False)
                ema.load_state_dict(ema_ckpt['ema_state_dict'])
                logger.info(f"Loaded EMA from {ema_path}")
        # Fast-forward LR scheduler
        for _ in range(start_epoch):
            lr_scheduler.step()
        logger.info(f"Resumed at epoch {start_epoch}")
    
    # max train steps
    num_update_steps_per_epoch = len(train_loader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    #  setup distributed training
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
        unet_wo_ddp = unet.module
        if class_embedder:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
            class_embedder_wo_ddp = class_embedder.module
    else:
        unet_wo_ddp = unet
        class_embedder_wo_ddp = class_embedder
    vae_wo_ddp = vae
    # TODO: setup ddim
    if args.use_ddim:
        scheduler_wo_ddp = DDIMScheduler(
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
        scheduler_wo_ddp = scheduler

    # TODO: setup evaluation pipeline
    # NOTE: this pipeline is not differentiable and only for evaluatin
    pipeline = DDPMPipeline(unet_wo_ddp, scheduler_wo_ddp, vae=vae_wo_ddp, class_embedder=class_embedder_wo_ddp)
    
    
    # dump config file
    if is_primary(args):
        experiment_config = vars(args)
        with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
    
    # start tracker
    if is_primary(args):
        wandb_logger = wandb.init(
            project='ddpm', 
            name=args.run_name, 
            config=vars(args))
    
    # Start training    
    if is_primary(args):
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_primary(args))

    # training
    for epoch in range(start_epoch, args.num_epochs):
        
        # set epoch for distributed sampler, this is for distribution training
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        args.epoch = epoch
        if is_primary(args):
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        
        loss_m = AverageMeter()
        
        # TODO: set unet and scheduelr to train
        unet.train()
        if class_embedder is not None:
            class_embedder.train()


        # TODO: finish this
        for step, (images, labels) in enumerate(train_loader):

            batch_size = images.size(0)

            # TODO: send to device
            images = images.to(device)
            labels = labels.to(device)


            # NOTE: this is for latent DDPM
            if vae is not None:
                # use vae to encode images as latents
                images = vae.encode(images)
                # NOTE: do not change  this line, this is to ensure the latent has unit std
                images = images * 0.1845

            # TODO: zero grad optimizer
            optimizer.zero_grad()

            # NOTE: this is for CFG
            if class_embedder is not None:
                # TODO: use class embedder to get class embeddings
                class_emb = class_embedder(labels)
            else:
                # NOTE: if not cfg, set class_emb to None
                class_emb = None

            # TODO: sample noise
            noise = torch.randn_like(images)

            # TODO: sample timestep t
            timesteps = torch.randint(0, args.num_train_timesteps, (batch_size,), device=device).long()

            # TODO: add noise to images using scheduler
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            # Forward pass (bf16 autocast on H100/H200 gives ~2-3x speedup)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                # TODO: model prediction
                raw_model_pred = unet(noisy_images, timesteps, class_emb)

                # Split learned-variance coefficients off before computing L_simple.
                if args.variance_type == 'learned_range':
                    model_pred, var_coef = torch.split(raw_model_pred, args.unet_in_ch, dim=1)
                else:
                    model_pred, var_coef = raw_model_pred, None

                if args.prediction_type == 'epsilon':
                    target = noise
                elif args.prediction_type == 'v_prediction':
                    target = scheduler.get_velocity(images, noise, timesteps)
                else:
                    raise NotImplementedError(f"Unknown prediction_type: {args.prediction_type}")

                if min_snr_gamma > 0:
                    loss_per_sample = ((model_pred.float() - target.float()) ** 2).mean(dim=list(range(1, model_pred.ndim)))
                    alpha_cumprod_t = scheduler.alphas_cumprod[timesteps].to(loss_per_sample.dtype)
                    snr = alpha_cumprod_t / (1.0 - alpha_cumprod_t)
                    # Min-SNR-gamma (Hang et al. 2023): clamp SNR weight to gamma for low-t stability
                    weight = torch.clamp(snr, max=min_snr_gamma) / snr
                    loss = (weight * loss_per_sample).mean()
                else:
                    loss = F.mse_loss(model_pred, target)

                # Hybrid loss: L_simple + λ · L_vlb. VLB only trains the variance head (eps is detached
                # inside scheduler.vlb_terms) — without this gate the VLB KL would overwhelm L_simple.
                if var_coef is not None and args.vlb_weight > 0:
                    vlb = scheduler.vlb_terms(images.float(), noisy_images.float(), timesteps,
                                              eps_pred=model_pred.float(), var_coef=var_coef.float())
                    loss = loss + args.vlb_weight * vlb.mean()

            # record loss
            loss_m.update(loss.item())

            # backward and step
            loss.backward()
            # TODO: grad clip
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)

            # TODO: step your optimizer
            optimizer.step()

            # Update EMA weights after each optimizer step
            if ema is not None:
                ema.update(unet_wo_ddp)

            progress_bar.update(1)
            
            # logger
            if step % 100 == 0 and is_primary(args):
                logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Step {step}/{num_update_steps_per_epoch}, Loss {loss.item()} ({loss_m.avg})")
                wandb_logger.log({'loss': loss_m.avg})

        # step the LR scheduler
        lr_scheduler.step()

        # validation
        # send unet to evaluation mode; swap in EMA weights for sampling if enabled
        unet.eval()
        if ema is not None:
            pipeline.unet = ema.ema_model        
        generator = torch.Generator(device=device)
        generator.manual_seed(epoch + args.seed)
        
        # NOTE: this is for CFG
        if args.use_cfg:
            # random sample 4 classes
            classes = torch.randint(0, args.num_classes, (4,), device=device)
            # TODO: fill pipeline
            gen_images = pipeline(
                batch_size=4,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device,
            )
        else:
            # TODO: fill pipeline
            gen_images = pipeline(
                batch_size=4,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )
            
        # create a blank canvas for the grid
        grid_image = Image.new('RGB', (4 * args.image_size, 1 * args.image_size))
        # paste images into the grid
        for i, image in enumerate(gen_images):
            x = (i % 4) * args.image_size
            y = 0
            grid_image.paste(image, (x, y))
        
        # Send to wandb
        if is_primary(args):
            wandb_logger.log({'gen_images': wandb.Image(grid_image)})

        # restore the raw UNet reference on the pipeline for next epoch's EMA swap
        if ema is not None:
            pipeline.unet = unet_wo_ddp

        # save checkpoint (including EMA weights if enabled). Gated by --save_every
        # to avoid filling disk on long runs; the final epoch is always saved.
        if is_primary(args):
            is_last = (epoch == args.num_epochs - 1)
            if is_last or ((epoch + 1) % args.save_every == 0):
                save_checkpoint(unet_wo_ddp, scheduler_wo_ddp, vae_wo_ddp, class_embedder, optimizer, epoch, save_dir=save_dir)
                if ema is not None:
                    ema_path = os.path.join(save_dir, f'ema_epoch_{epoch}.pth')
                    torch.save({'ema_state_dict': ema.state_dict(), 'epoch': epoch}, ema_path)


if __name__ == '__main__':
    main()
