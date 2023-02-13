from diffusers import DDPMPipeline
from PIL import Image
import math
import torch
import os
from defs import DEVICE, EVAL_BATCH_SIZE, SAMPLE_DIR, SEED
from minepi import Skin
from PIL import Image
import time


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


async def render_skin(image):
    s = Skin(raw_skin=image)
    await s.render_skin()
    return s.skin


def evaluate(epoch, pipeline, size=EVAL_BATCH_SIZE):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    generator = torch.Generator(device=DEVICE)
    generator = generator.manual_seed(SEED)
    images = pipeline(
        batch_size=size,
        generator=generator,
    ).images
    os.makedirs(f"{SAMPLE_DIR}/{epoch:04d}", exist_ok=True)
    print(len(images))
    for idx, i in enumerate(images):
        i.save(f"{SAMPLE_DIR}/{epoch:04d}/{idx}.png")
    # # Make a grid out of the images
    # image_grid = make_grid(
    #     images,
    #     rows=int(size ** (1 / 2)),
    #     cols=int(size ** (1 / 2)),
    # )

    # # Save the images
    # os.makedirs(SAMPLE_DIR, exist_ok=True)
    # image_grid.save(f"{SAMPLE_DIR}/{epoch:04d}.png")
