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
    import os
    import sys
    import torch
    from safetensors import safe_open

     # Allow animatediff import and calls
    os.chdir('/data/repos/animatediff/')
    sys.path.append('/data/repos/animatediff/')

     # Import animatediff
    from animatediff.utils.util import save_videos_grid
    from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint
    from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora

    # Load cached pipeline
    pipeline = model()

     # Other params
    lora_alpha=0.8
    base=""
    full_path = "models/DreamBooth_LoRA/toonyou_beta3.safetensors"

    # Load motion model
    motion_path = "models/Motion_Module/mm_sd_v14.ckpt"
    motion_module_state_dict = torch.load(motion_path, map_location="cpu")
    missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
    assert len(unexpected) == 0

    state_dict = {}
    base_state_dict = {}
    with safe_open(full_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    is_lora = all("lora" in k for k in state_dict.keys())
    if not is_lora:
        base_state_dict = state_dict
    else:
        base_state_dict = {}
        with safe_open(base, framework="pt", device="cpu") as f:
            for key in f.keys():
                base_state_dict[key] = f.get_tensor(key)
    # vae
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
    pipeline.vae.load_state_dict(converted_vae_checkpoint)
    # unet
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
    # lora
    if is_lora:
        pipeline = convert_lora(pipeline, state_dict, alpha=lora_alpha)

    pipeline.to("cuda")

    outname = "output.gif"
    outpath = f"./{outname}"
    sample = pipeline(
        prompt,
        negative_prompt     = "",
        num_inference_steps = 25,
        guidance_scale      = 7.5,
        width               = 512,
        height              = 512,
        video_length        = 16,
    ).videos
    samples = torch.concat([sample])
    save_videos_grid(samples, outpath , n_rows=1)
    output_data = None
    with open(outpath, "rb") as file:
        output_data = file.read()

    return output_data


    # return the output that you want to be returned like pre-signed URLs to output artifacts
    # return "Hello World"


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

prompt = "masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, plum blossoms, outdoors, falling petals, white hair, black eyes"