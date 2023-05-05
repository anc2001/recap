import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os
from PIL import Image
import numpy as np
import pickle as pkl

class FlickrDataset(Dataset):
    def __init__(self, base_filepath, transform = None):
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
        self.generated_img_folder = os.path.join(base_filepath, "generated_images")
        self.transform = transform
        self.annotations = dict(zip(annotations[0], annotations[1]))

        # self.img_filepaths = list(human_annotations[0])
        # self.caption_ids = list(human_annotations[1])
        # self.human_scores = list(human_annotations[2])
        
        self.img_filepaths = list(expert[0])
        self.caption_ids = list(expert[1])
        self.human_scores = list(expert['score'])

        with open("../data/splits.pkl", 'rb') as handle:
            splits = pkl.load(handle)
        
        self.splits = splits
        
    def __len__(self):
        return len(self.img_filepaths)

    def __getitem__(self, idx : int):
        img_filepath = self.img_filepaths[idx]
        caption_id = self.caption_ids[idx]
        human_score = self.human_scores[idx]
        caption = self.annotations[caption_id]
        img = read_image(os.path.join(self.img_folder, img_filepath))
        if self.transform:
            img = self.transform(img)

        # generated_img = read_image(os.path.join(self.generated_img_folder, caption_id))
        # if self.transform:
        #     img = self.transform(img)

        return img, caption, human_score

