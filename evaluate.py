from diffusers import DDPMPipeline
from PIL import Image
import math
import torch
import os
from defs import DEVICE, EVAL_BATCH_SIZE, SAMPLE_DIR, SEED


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=EVAL_BATCH_SIZE,
        generator=torch.Generator(device=DEVICE),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(
        images,
        rows=int(EVAL_BATCH_SIZE ** (1 / 2)),
        cols=int(EVAL_BATCH_SIZE ** (1 / 2)),
    )

    # Save the images
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    image_grid.save(f"{SAMPLE_DIR}/{epoch:04d}.png")
