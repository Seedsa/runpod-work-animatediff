
def clone_repo():
    import os
    os.system("apt-get update && apt-get install -y git-lfs")
    os.system("git lfs install")
    repo_path = "/data/repos/animatediff/"
    if not os.path.exists(repo_path):
        print("Cloning AnimateDiff repository")
        os.system(
            f"git clone https://github.com/guoyww/AnimateDiff {repo_path}")
        print("Done")


def clone_stable_diff():
    import os
    import sys
    os.chdir('/data/repos/animatediff/')
    sys.path.append('/data/repos/animatediff/')
    repo_path = "models/StableDiffusion/stable-diffusion-v1-5"
    if not os.path.exists(repo_path):
        print("Cloning StableDiffusionv1.5")
        os.system("rm -rf models/StableDiffusion/stable-diffusion-v1-5")
        os.system(f"git clone --branch fp16 https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/stable-diffusion-v1-5")
        print("Done")


def download_motion_module():
    import os
    model_path = "/data/repos/animatediff/models/Motion_Module/"
    if os.path.exists(model_path):
        print("Downloading model: mm_sd_v14")
        url = "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt"
        url2 = "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt"
        os.system(f"cd {model_path} && wget {url} && wget {url2}")
        print("Done")


def download_dreambooth_lora():
    import os
    model_path = "/data/repos/animatediff/models/DreamBooth_LoRA/"
    if os.path.exists(model_path):
        print("Downloading model: toonyou_beta3")
        url = "https://civitai.com/api/download/models/78775"
        os.system(
            f"cd {model_path} && wget -O toonyou_beta3.safetensors {url}")
        print("Done")


if __name__ == "__main__":
    clone_repo()
    clone_stable_diff()
    download_motion_module()
    download_dreambooth_lora()
