import os
import subprocess

BASE_PATH = "/data/repos/animatediff/"

REPO_PATHS = {
    "animate_diff": os.path.join(BASE_PATH, "repos", "animatediff"),
    "stable_diff": os.path.join(BASE_PATH, "models", "StableDiffusion", "stable-diffusion-v1-5"),
    "motion_module": os.path.join(BASE_PATH, "models", "Motion_Module"),
    "dreambooth_lora": os.path.join(BASE_PATH, "models", "DreamBooth_LoRA"),
}

DOWNLOADS = {
    "motion_module": [
        "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt",
        "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt"
    ],
    "dreambooth_lora": [
        ("https://civitai.com/api/download/models/78775", "toonyou_beta3.safetensors")
    ]
}


def run_command(cmd, directory=None):
    try:
        subprocess.check_call(cmd, cwd=directory, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


def clone_repo():
    if not os.path.exists(REPO_PATHS["animate_diff"]):
        print("Cloning AnimateDiff repository")
        run_command(
            f"git clone https://github.com/guoyww/AnimateDiff {REPO_PATHS['animate_diff']}")
        print("Done")


def clone_stable_diff():
    os.chdir(REPO_PATHS["animate_diff"])

    if not os.path.exists(REPO_PATHS["stable_diff"]):
        print("Cloning StableDiffusionv1.5")
        run_command("rm -rf models/StableDiffusion/stable-diffusion-v1-5")
        run_command(
            f"git clone --branch fp16 https://huggingface.co/runwayml/stable-diffusion-v1-5 {REPO_PATHS['stable_diff']}")
        print("Done")


def download_files(model_key):
    for url in DOWNLOADS[model_key]:
        if isinstance(url, tuple):
            url, filename = url
            print(f"Downloading model: {filename}")
            run_command(f"wget -O {filename} {url}", REPO_PATHS[model_key])
        else:
            print(f"Downloading model from: {url}")
            run_command(f"wget {url}", REPO_PATHS[model_key])
        print("Done")


def main():
    run_command("apt-get update && apt-get install -y git-lfs")
    run_command("git lfs install")

    clone_repo()
    clone_stable_diff()
    download_files("motion_module")
    download_files("dreambooth_lora")


if __name__ == "__main__":
    main()
