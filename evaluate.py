from PIL import Image
import torch
import os
from defs import DEVICE, EVAL_BATCH_SIZE, SAMPLE_DIR, SEED
from PIL import Image


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


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
    for idx, i in enumerate(images):
        i.save(f"{SAMPLE_DIR}/{epoch:04d}/{idx}.png")
