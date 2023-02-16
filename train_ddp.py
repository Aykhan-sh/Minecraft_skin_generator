import albumentations as A
from dataset import CoverDataset
from torch.utils.data import DataLoader
from evaluate import evaluate
from models import get_model
import albumentations as A
import os
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from defs import (
    IMSIZE,
    SAMPLE_DIR,
    TRAIN_BATCH_SIZE,
    DATA_PATH,
    LR,
    NUM_TRAIN_TIMESTEPS,
    EPOCHS,
    DEVICE,
    EVALUATION_INTERVAL,
    WEIGHTS_DIR,
    USE_RAM,
    MIXED_PRECISION,
    NUM_WORKERS,
)

torch.backends.cudnn.benchmark = True
torch.jit.enable_onednn_fusion(True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_dataloader(
    rank, world_size, batch_size=32, pin_memory=False, num_workers=4
):
    transforms = None
    dataset = CoverDataset(DATA_PATH, IMSIZE, transforms, True, USE_RAM)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        sampler=sampler,
    )


def train(rank, world_size):
    setup(rank, world_size)
    dataloader = prepare_dataloader(rank, world_size, TRAIN_BATCH_SIZE, False, 4)
    model = get_model(IMSIZE).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    device = torch.device("cuda", rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS, beta_schedule="squaredcos_cap_v2"
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(len(dataloader) * 0.1),
        num_training_steps=(len(dataloader) * EPOCHS),
    )

    for epoch in range(EPOCHS):
        train_loss_mean = torch.tensor(0, dtype=float, device=device)
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        dataloader.sampler.set_epoch(epoch)
        model.train()
        for step, imgs in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            imgs = imgs.to(device).float()  # preprocess
            imgs = (imgs - 0.5) * 2
            noise = torch.randn(imgs.shape).to(device)
            timesteps = torch.randint(
                0,
                NUM_TRAIN_TIMESTEPS,
                (TRAIN_BATCH_SIZE,),
                device=device,
                dtype=torch.long,
            )
            noisy_images = noise_scheduler.add_noise(imgs, noise, timesteps)
            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
            if MIXED_PRECISION:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(loss)
                optimizer.step()

            # clip_grad_norm_(model.parameters(), 1.0)
            lr_scheduler.step()
            progress_bar.update(1)
            train_loss_mean += loss
        if rank == 0:
            print()
            print(f"Epoch {epoch}")
            print(f"Loss: {(train_loss_mean / len(dataloader)).item()}")
            print(f"Lr: {lr_scheduler.get_last_lr()[0]}")
            model.eval()
            pipeline = DDPMPipeline(unet=model.module, scheduler=noise_scheduler)
            if (epoch + 1) % EVALUATION_INTERVAL == 0 or epoch == EPOCHS - 1:
                evaluate(epoch, pipeline)
                pipeline.save_pretrained(WEIGHTS_DIR)
    cleanup()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size)
