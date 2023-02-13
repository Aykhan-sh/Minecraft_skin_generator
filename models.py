from diffusers import UNet2DModel

_models_list = ["CCMat/ddpm-bored-apes-128", "google/ddpm-celebahq-256"]


def get_models(imsize: int):
    # return UNet2DModel.from_pretrained("CCMat/ddpm-bored-apes-128")
    unet = UNet2DModel(
        sample_size=imsize,  # the target image resolution
        in_channels=4,  # the number of input channels, 3 for RGB images
        out_channels=4,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(
            128,
            128,
            256,
            256,
            512,
            512,
        ),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return unet
