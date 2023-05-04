import os
import io
import time
import argparse
import warnings

from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

with open("secret.txt", 'r') as f:
	key = f.readline().strip()

# https://platform.stability.ai/docs/getting-started/python-sdk
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = key 

def main(flags):
    # Make the image size the same as the dataset, retrieve prompts
    if flags.dataset == "flickr":
        img_size = None # TODO
        prompts = [] # TODO

    # set up stability api connection 
    # https://github.com/Stability-AI/api-interfaces/blob/main/src/proto/generation.proto
    stability_api = client.StabilityInference(
        key=os.environ['STABILITY_KEY'],
        verbose=True,
        engine="stable-diffusion-xl-beta-v2-2-2",
    )
    generate_params = { 
        "seed" : 2952, 
        "steps" : 30,
        "cfg_scale" : 8.0,
        "width" : img_size,
        "height" : img_size, 
        "samples" : 2,
        "sampler" : generation.SAMPLER_K_DPMPP_2M
    }

    # For each prompt, generate images and save them.
    path = f"../data/generated_images/{flags.dataset}/"
    os.makedirs(path, exist_ok=True)
    for p_id, prompt in enumerate(prompts):
        responses = stability_api.generate(
            prompt=prompt,
            **generate_params
        )

        # code from tutorial to save images
        for i, resp in enumerate(responses):
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed."
                        "Please modify the prompt and try again.")
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    img.save(path + f"prompt{p_id}_{i}.png") 


if __name__ == "__main__":
    tick = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["flickr"], required=True)
    flags = parser.parse_args()
  
    main(flags)

    tock = time.time()
    print(tock - tick, "seconds")