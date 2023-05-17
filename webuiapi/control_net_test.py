import numpy as np
import torch
import transformers.utils
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

input_image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

# load the pipeline dynamically
pipe_controlnet = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
    custom_pipeline="stable_diffusion_controlnet_img2img")

pipe_controlnet.scheduler = UniPCMultistepScheduler.from_config(pipe_controlnet.scheduler.config)
# pipe_controlnet.enable_xformers_memory_efficient_attention()
pipe_controlnet.enable_model_cpu_offload()

# using image with edges for our canny controlnet
control_image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/vermeer_canny_edged.png")
control_image.show()

result_img = pipe_controlnet(controlnet_conditioning_image=control_image,
                        image=input_image,
                        prompt="cute little girl face, anime, high resolution",
                        num_inference_steps=20).images[0]
result_img.show()

