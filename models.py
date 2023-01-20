from diffusers import UNet2DModel

_models_list = ["CCMat/ddpm-bored-apes-128", "google/ddpm-celebahq-256"]


def get_models(imsize: int):
    return UNet2DModel.from_pretrained("CCMat/ddpm-bored-apes-128")
