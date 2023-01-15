import albumentations as A
from dataset import CoverDataset
from torch.utils.data import DataLoader
from evaluate import evaluate
from models import get_models
import albumentations as A
import os
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
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
)

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

transforms = None
dataset = CoverDataset(DATA_PATH, IMSIZE, transforms, True, USE_RAM)
train_dataloader = DataLoader(
    dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
)

# model
model = get_models(IMSIZE).to(DEVICE)

# utils
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=len(train_dataloader),
    num_training_steps=(len(train_dataloader) * EPOCHS),
)


for epoch in tqdm(range(EPOCHS)):
    progress_bar = tqdm(total=len(train_dataloader))
    progress_bar.set_description(f"Epoch {epoch}")
    for step, imgs in enumerate(train_dataloader):
        imgs = imgs.to(DEVICE).float()
        noise = torch.randn(imgs.shape).to(DEVICE)
        timesteps = torch.randint(
            0, NUM_TRAIN_TIMESTEPS, (TRAIN_BATCH_SIZE,), device=DEVICE
        ).long()
        noisy_images = noise_scheduler.add_noise(imgs, noise, timesteps)

        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)

        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
        }
        progress_bar.set_postfix(**logs)

    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    if (epoch + 1) % EVALUATION_INTERVAL == 0 or epoch == EPOCHS - 1:
        evaluate(epoch, pipeline)
        pipeline.save_pretrained(WEIGHTS_DIR)
