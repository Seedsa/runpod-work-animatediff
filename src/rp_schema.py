INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'width': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda width: width in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'video_length': {
        'type': int,
        'required': False,
        'default': 16,
    },
    'height': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda height: height in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 25,
        'constraints': lambda num_inference_steps: 0 < num_inference_steps < 500
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.5,
        'constraints': lambda guidance_scale: 0 < guidance_scale < 20
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'DDIM',
        'constraints': lambda scheduler: scheduler in ['DDIM', 'DDPM', 'DPM-M', 'DPM-S',  'EULER-A', 'EULER-D', 'HEUN', 'IPNDM', 'KDPM2-A', 'KDPM2-D', 'PNDM', 'K-LMS', 'KLMS']
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'lora': {
        'type': str,
        'required': False,
        'default': None
    },
    'lora_scale': {
        'type': float,
        'required': False,
        'default': 1,
        'constraints': lambda lora_scale: 0 <= lora_scale <= 1
    }
}
