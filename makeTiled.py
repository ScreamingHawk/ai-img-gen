import torch
import re
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import safety_checker

# Override NFSW filter
def sc(self, clip_input, images) :
    return images, [False for i in images]
safety_checker.StableDiffusionSafetyChecker.forward = sc

# Tile
def patch_conv(cls):
  init = cls.__init__
  def __init__(self, *args, **kwargs):
    return init(self, *args, **kwargs, padding_mode='circular')
  cls.__init__ = __init__
patch_conv(torch.nn.Conv2d)

seed = 69420
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
# pipe.to("cuda")

prompts = [
	"fish made out of spagitti",
]

for prompt in prompts:
	for i in range(1, 10):
		print(prompt, seed)
		generator = torch.Generator().manual_seed(seed) # Seeded random
		seed += 1
		image = pipe(prompt, generator=generator).images[0]
		image.save(f"{re.sub('[^0-9a-zA-Z]+', '_', prompt)}_{seed}_tile.png")
