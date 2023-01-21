from diffusers import UNet2DModel

_models_list = ["CCMat/ddpm-bored-apes-128", "google/ddpm-celebahq-256"]


def get_models(imsize: int):
    unet = UNet2DModel.from_pretrained("google/ddpm-celebahq-256")
    unet.sample_size = imsize
    return unet
