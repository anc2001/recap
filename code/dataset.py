import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
from PIL import Image
import numpy as np
import pickle as pkl

class FlickrDataset(Dataset):
    def __init__(
            self, base_filepath, 
            type = "annotated", transform = None, swap = False, generated_img_tag=''
        ):
        self.transform = transform
        self.type = type
        self.swap = swap
        self.generated_img_tag = generated_img_tag
        annotation_filepath = os.path.join(base_filepath, "Flickr8k_text")
        expert = pd.read_csv(
            os.path.join(annotation_filepath, "ExpertAnnotations.txt"),
            sep='\t',
            header=None
        )
        # crowd = pd.read_csv(
        #     os.path.join(annotation_filepath, "CrowdFlowerAnnotations.txt"),
        #     sep='\t',
        #     header=None
        # )
        annotations = pd.read_csv(
            os.path.join(annotation_filepath, "Flickr8k.token.txt"),
            sep='\t',
            header=None
        )

        expert['score'] = expert[[2, 3, 4]].mean(axis=1) / 4
        expert = expert.drop(columns=[2, 3, 4])
        # crowd['score'] = crowd[2]
        # crowd = crowd.drop(columns=[2, 3, 4])

        # human_annotations = pd.concat([crowd, expert])
        # human_annotations.sort_values('score').drop_duplicates(subset=[0, 1], keep='last')

        self.img_folder = os.path.join(base_filepath, "Flicker8k_Dataset")
        self.generated_img_folder = os.path.join(base_filepath, f"generated_images{generated_img_tag}")
        self.annotations = dict(zip(annotations[0], annotations[1]))

        # self.img_filepaths = list(human_annotations[0])
        # self.caption_ids = list(human_annotations[1])
        # self.human_scores = list(human_annotations[2])
        
        self.caption_ids = list(expert[1])
        self.human_scores = list(expert['score'])

        if self.type == 'annotated':
            self.img_filepaths = list(expert[0])
        elif self.type == 'matching':
            self.img_filepaths = [caption_id.split("#")[0] for caption_id in self.caption_ids]
        else:
            print(f"{self.type} is invalid")
            exit()

        with open("../data/splits.pkl", 'rb') as handle:
            splits = pkl.load(handle)
        
        self.splits = splits
        
    def __len__(self):
        return len(self.img_filepaths)

    def __getitem__(self, idx : int):
        img_filepath = self.img_filepaths[idx]
        if self.swap:
            new_img_filepath = np.random.choice(self.img_filepaths)
            while new_img_filepath == img_filepath:
                new_img_filepath = np.random.choice(self.img_filepaths)
            img_filepath = new_img_filepath
        
        caption_id = self.caption_ids[idx]
        human_score = self.human_scores[idx]
        img = Image.open(os.path.join(self.img_folder, img_filepath))
        if self.transform:
            img = self.transform(img)
        
        generated_imgs = []
        for i in range(3):
            generated_img = Image.open(
                os.path.join(
                    self.generated_img_folder, f"{caption_id}{self.generated_img_tag}_{i}.png"
                )
            )
            if self.transform:
                generated_img = self.transform(generated_img)
            if len(generated_imgs):
                generated_imgs = torch.concat(
                    [
                        generated_imgs, 
                        torch.unsqueeze(generated_img, 0)
                    ],
                    dim = 0
                )
            else:
                generated_imgs = torch.unsqueeze(generated_img, 0)
        generated_imgs = generated_imgs
        
        if self.type == 'annotated':
            return img_filepath, img, caption_id, human_score, generated_imgs
        else:
            return img_filepath, img, caption_id, generated_imgs
        