
import torch
from diffusers import DiffusionPipeline


from huggingface_hub import notebook_login
notebook_login()

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

prompt = "An astronaut riding a green horse"
images = pipe(prompt=prompt).images[0]


from PIL import Image
from IPython.display import display
display(images)
