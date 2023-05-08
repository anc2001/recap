import pickle as pkl
from tqdm import tqdm
from dataset import FlickrDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import piq
import torch
import time
import argparse
import pandas as pd

img_size = 256
bsz = 16
num_generated_imgs = 3

def evaluate_metric(metric, imgs, generated_imgs):
    """
    metric: PIQ metric class 
    imgs: [N, 3, H, W]
    generated_imgs: [N, NUM_IMGS, 3, H, W]

    returns
    loss: [N, 3] - score for each image in the batch 
    base_scores: [N] - internal score 
    """
    base_score = []
    for item in [(0,1), (0,2), (1,2)]:
        img_1 = generated_imgs[:, item[0]]
        img_2 = generated_imgs[:, item[1]]
        base_loss = metric(img_1, img_2)
        if len(base_score):
            base_score += base_loss
        else:
            base_score = base_loss
    base_score /= 3
    
    generated_imgs_flat = generated_imgs.view(bsz * num_generated_imgs, 3, img_size, img_size)
    imgs_flat = torch.unsqueeze(imgs, 1).expand(-1, 3, 3, img_size, img_size). \
        reshape(bsz * num_generated_imgs, 3, img_size, img_size) / 255
    loss = metric(imgs_flat, generated_imgs_flat).view(-1, 3)

    return loss, base_score

metrics = {
    "SSIM" : piq.SSIMLoss(reduction='none')
}

def main(flags):
    metric = metrics[flags.metric]
    transform = transforms.Resize([img_size, img_size], antialias=True)
    to_pil = transforms.ToPILImage()
    dataset = FlickrDataset("../data", transform)

    # create a new csv for each metric as a list  
    cols = ['caption_id', 'generated_image', f'max_{flags.metric}', f'avg_{flags.metric}']
    out = []

    dataloader = DataLoader(dataset, batch_size = bsz, shuffle=True)
    for imgs, captions, human_scores, generated_imgs in tqdm(dataloader):
        loss, base_score = evaluate_metric(metric, imgs, generated_imgs)
        max_score = torch.max(loss, axis=1)[0]

        # TODO
        out.append([caption_id, generated_image])

    df_out = pd.DataFrame(out, columns=cols)
    assert(len(df_out) == 2931)
    # TODO save csv

if __name__ == '__main__':
    tick = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=metrics.keys(), required=True)
    flags = parser.parse_args()
  
    main(flags)

    tock = time.time()
    print(tock - tick, "seconds")
