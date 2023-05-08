import os
import io
import time
import random
import argparse
import warnings

from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
from torchvision import transforms
from dataset import FlickrDataset

def main(flags):
    # Make the image size the same as the dataset, retrieve prompts
    if flags.dataset == "flickr":
        img_size = 256 
        dataset = FlickrDataset("../data")
        if flags.split in [0,1,2,3]:
            caption_ids = dataset.splits[flags.split]
        else:
            caption_ids = list(set(dataset.caption_ids))
    
    model_id = "stabilityai/stable-diffusion-2"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    if flags.shuffle:
        os.makedirs("../data/generated_images_shuf")

    num_imgs = 3
    for caption_id in tqdm(caption_ids):
        prompt = dataset.annotations[caption_id]
        prompt = prompt.strip('.').strip(' ')
        if flags.shuffle:
            temp = prompt.split(' ')
            random.shuffle(temp)
            prompt = ' '.join(temp)
        prompt += ", color photo"
        for i in range(num_imgs):
            if flags.shuffle:
                filepath = f"../data/generated_images_shuf/{caption_id}_{i}.png"
            else:
                filepath = f"../data/generated_images/{caption_id}_{i}.png"
            print(filepath)
            if not os.path.isfile(filepath):
                generator = torch.Generator("cuda").manual_seed(i)
                image = pipe(prompt=prompt, generator=generator, num_inference_steps=50).images[0]
                image.save(filepath)
            else:
                print(f"{filepath} exists -> skipped!")

if __name__ == "__main__":
    tick = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["flickr"], required=True)
    parser.add_argument("--split", choices=[0,1,2,3], type=int, default=-1)
    parser.add_argument("--shuffle", action='store_true')
    parser.set_defaults(shuffle=False)
    flags = parser.parse_args()
  
    main(flags)

    tock = time.time()
    print(tock - tick, "seconds")
