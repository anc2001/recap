import os
import io
import time
import argparse
import warnings

from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
from torchvision import transforms
from dataset import FlickrDataset


def main(flags):
    # Make the image size the same as the dataset, retrieve prompts
    if flags.dataset == "flickr":
        img_size = 256 
        transform = transforms.Resize([img_size, img_size])
        dataset = FlickrDataset("../data", transform)
        prompts = dataset.annotations
    
    model_id = "stabilityai/stable-diffusion-2"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    num_imgs = 4
    for p_id, prompt in prompts.items():
        for i in range(num_imgs):
            generator = torch.Generator("cuda").manual_seed(i)
            image = pipe(prompt=prompt, generator=generator).images[0]
            image.save(f"../data/generated_images/{p_id}_{i}.png")

if __name__ == "__main__":
    tick = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["flickr"], required=True)
    flags = parser.parse_args()
  
    main(flags)

    tock = time.time()
    print(tock - tick, "seconds")