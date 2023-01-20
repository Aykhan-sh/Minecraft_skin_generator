from typing import Literal

IMSIZE = 128
DEVICE = "cuda"
DATA_PATH = "data/album_covers_512/"
SAMPLE_DIR = "results/samples"
WEIGHTS_DIR = "results/weights"
SEED = 23
EVALUATION_INTERVAL = 1
USE_RAM = False
PRECISION: Literal["fp16", "no"] = "fp16"

TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 16
LR = 0.0001
NUM_TRAIN_TIMESTEPS = 1000
EPOCHS = 1000
