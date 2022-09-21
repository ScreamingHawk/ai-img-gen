import torch
import re
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import safety_checker

# Override NFSW filter
def sc(self, clip_input, images) :
    return images, [False for i in images]
safety_checker.StableDiffusionSafetyChecker.forward = sc

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
generator = torch.Generator().manual_seed(69420) # Seeded random

prompts = [
	"Brown kitten with white fur",
	"Person walking in the city",
	"House with two layers and four bedrooms",
	"Sunset over a beach",
	"Tree with blossoms",
	"Magical cat",
	"Velociraptor eating a person",
	"Tyrannosaurus eating a house",
	"Tyrannosaurus eating a brontosaurus",
	"Similing poop with sparkles",
]

for prompt in prompts:
	with torch.autocast("cuda"):
		print(prompt)
		image = pipe(prompt, generator=generator).images[0]
		image.save(f"{re.sub('[^0-9a-zA-Z]+', '_', prompt)}.png")
