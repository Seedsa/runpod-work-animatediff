#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless. '''

import runpod

# Load models into VRAM here so they can be warm between requests


def model():
    import os
    import sys
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers.utils.import_utils import is_xformers_available

    os.chdir('/data/repos/animatediff/')
    sys.path.append('/data/repos/animatediff/')
    from animatediff.models.unet import UNet3DConditionModel
    from animatediff.pipelines.pipeline_animation import AnimationPipeline

    # Load config
    inference_config_file = "configs/inference/inference.yaml"
    inference_config = OmegaConf.load(inference_config_file)
    pretrained_model_path = "models/StableDiffusion/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            inference_config.unet_additional_kwargs)
    )
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    pipeline = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler(
            **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
    )
    pipeline = pipeline.to("cuda")
    return pipeline


def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    print(event)

    # do the things

    # return the output that you want to be returned like pre-signed URLs to output artifacts
    return "Hello World"


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
