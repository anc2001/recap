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

img_size = 256
bsz = 16
num_generated_imgs = 3

def evaluate_metric(metric, imgs, generated_imgs):
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

    dataloader = DataLoader(dataset, batch_size = bsz, shuffle=True)
    for imgs, captions, human_scores, generated_imgs in tqdm(dataloader):
        loss, base_score = evaluate_metric(metric, imgs, generated_imgs)
        max_score = torch.max(loss, axis=0)
        break
    print(loss.shape, base_score.shape, max_score.shape, bsz)

if __name__ == '__main__':
    tick = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=metrics.keys(), required=True)
    flags = parser.parse_args()
  
    main(flags)

    tock = time.time()
    print(tock - tick, "seconds")