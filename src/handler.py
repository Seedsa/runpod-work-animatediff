#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless. '''

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_upload
from runpod.serverless.utils import rp_download, upload_file_to_bucket
from rp_schema import INPUT_SCHEMA
import uuid

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


def handler(job):
    '''
    This is the handler function that will be called by the serverless.
    '''
    job_input = job['input']

    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {'error': job_input['errors']}
    job_input = job_input['validated_input']
    print('input:',job_input)
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
    full_path = f"models/DreamBooth_LoRA/{job_input['base_model']}"

    # Load motion model
    motion_path = f"models/Motion_Module/{job_input['motion_model']}"
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
    uid = uuid.uuid4()
    outname = f"{uid}.gif"
    outpath = f"./{outname}"
    sample = pipeline(
        prompt              = job_input['prompt'],
        negative_prompt     = job_input['negative_prompt'],
        num_inference_steps = job_input['steps'],
        guidance_scale      = job_input['guidance_scale'],
        width               = job_input['width'],
        height              = job_input['height'],
        video_length        = job_input['video_length'],
    ).videos
    samples = torch.concat([sample])
    save_videos_grid(samples, outpath , n_rows=1)
    uploaded_url = upload_file_to_bucket(
        file_name=f"{outname}",
        file_location=f"{outpath}",
        bucket_name=None if job_input['bucket_name'] is None else job_input['bucket_name']
    )
    return { "url": uploaded_url}
    # output_data = None
    # image_url = rp_upload.upload_image(job['id'], outpath)
    # with open(outpath, "rb") as file:
    #     output_data = file.read()

    # return output_data


    # return the output that you want to be returned like pre-signed URLs to output artifacts
    # return "Hello World"


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

