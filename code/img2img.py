import pickle as pkl
from tqdm import tqdm
from dataset import FlickrDatasetAnnotated, FlickrDatasetMatching
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import piq
import torch
import time
import argparse
import pandas as pd
import os
import numpy as np
from clip_score import CLIP_image_score

bsz = 16
num_generated_imgs = 3

def evaluate_metric(metric, imgs, generated_imgs):
    """
    metric: PIQ metric class 
    imgs: [N, 3, H, W]
    generated_imgs: [N, NUM_IMGS, 3, H, W]

    returns
    loss: [N, NUM_IMGS] - score for each image in the batch 
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

    print(generated_imgs.shape)
    
    img_size = imgs.size(2)
    generated_imgs_flat = generated_imgs.view(bsz * num_generated_imgs, 3, img_size, img_size)
    imgs_flat = imgs.unsqueeze(1).expand(-1, 3, 3, img_size, img_size). \
        reshape(bsz * num_generated_imgs, 3, img_size, img_size)
    loss = metric(imgs_flat, generated_imgs_flat).view(-1, num_generated_imgs)

    return loss, base_score

metrics = {
    "SSIM" : piq.SSIMLoss(reduction='none'),
    "MS-SSIM" : piq.MultiScaleSSIMLoss(reduction='none'),
    "VIF" : piq.VIFLoss(reduction='none'),
    "GMSD" : piq.GMSDLoss(reduction='none'),
    "FSIM" : piq.FSIMLoss(reduction='none'),
#    "CLIP_RN50" : CLIP_image_score("RN50"),
#    "CLIP_ViT-L/14" : CLIP_image_score("ViT-L/14")
}

datasets = {
    'matching' : FlickrDatasetMatching("../data"),
    'annotated' : FlickrDatasetAnnotated("../data"),
    'shuffled' : FlickrDatasetAnnotated("../data", generated_img_tag='_shuf')
}

def main(flags):
    metric = metrics[flags.metric]
    dataset = datasets[flags.dataset]

    if flags.metric == "CLIP_RN50" or flags.metric == "CLIP_ViT-L/14":
        transform = metric.preprocess
    else:
        transform = transforms.Compose([
            transforms.Resize([img_size, img_size], antialias=True),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
    dataset.transform = transform

    # create a new csv for each metric as a list  
    cols = ['img_id', 'caption_id', 'human_scores', 'generated_image_id', f'{flags.metric}', f'internal_baseline_{flags.metric}']
    out = []

    dataloader = DataLoader(dataset, batch_size = bsz, shuffle=True)
    for collated_vals in tqdm(dataloader):
        if flags.dataset == 'matching':
            (
                img_filepaths, imgs, caption_ids, generated_imgs
            ) = collated_vals
        elif flags.dataset in ['annotated', 'shuffled']:
            (
                img_filepaths, imgs, caption_ids, human_scores, generated_imgs
            ) = collated_vals

        if flags.noise:
            generated_imgs = torch.rand(generated_imgs.shape)
        scores, base_score = evaluate_metric(metric, imgs, generated_imgs)
        
        # caption_ids size [N]
        # human_scores size [N]
        # generated_image_ids size [N, NUM_IMGS] <- we have to make this
        # loss [N, NUM_IMGS]
        # base_score [N] 
        generated_img_ids = []
        for caption_id in caption_ids:
            for i in range(num_generated_imgs):
                if flags.dataset == "shuffled":
                    generated_img_ids.append(caption_id + f'_shuf_{i}')
                else:
                    generated_img_ids.append(caption_id + f'_{i}')
        
        col0 = np.expand_dims(np.repeat(np.array(img_filepaths), 3), 1)
        col1 = np.expand_dims(np.repeat(np.array(caption_ids), 3), 1)
        col2 = np.expand_dims(np.repeat(np.array(human_scores), 3), 1)
        col3 = np.expand_dims(generated_img_ids, 1)
        col4 = scores.view(-1, 1).numpy()
        col5 = np.expand_dims(np.repeat(np.array(base_score), 3), 1)
        new_rows = np.concatenate([col0, col1, col2, col3, col4, col5], axis=1)

        if len(out):
            out = np.append(out, new_rows, axis = 0)
        else:
            out = new_rows


    df_out = pd.DataFrame(out.tolist(), columns=cols)
    assert(len(df_out) == 2931)

    os.makedirs("../results", exist_ok=True)
    df_out.to_csv(f"../results/{flags.metric}_{flags.dataset}" + \
                  f"{'_shuf' if flags.dataset == 'shuffled' else ''}" + \
                  f"{'_noise' if flags.noise else ''}.csv", sep='\t')

if __name__ == '__main__':
    tick = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=datasets.keys(), required=True)
    parser.add_argument("--metric", choices=metrics.keys(), required=True)
    parser.add_argument("--noise", action='store_true')
    parser.set_defaults(shuffled=False)
    flags = parser.parse_args()
  
    main(flags)

    tock = time.time()
    print(tock - tick, "seconds")
