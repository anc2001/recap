import os
import argparse
import numpy as np
import pandas as pd
from img2img import metrics

def subsample(modification, base, indices, split_on):
    scores = np.append(
        modification[indices[:split_on]], 
        base[indices[split_on:]]
    )
    return np.mean(scores)

def main(flags):
    # for metric in metrics.keys():
    for metric in ["CLIP_RN50"]:
        print(metric)
        matching = pd.read_csv(f"../results/{metric}_matching.csv", sep='\t', index_col=0)
        matching_swapped = pd.read_csv(f"../results/{metric}_matching_swapped.csv", sep='\t', index_col=0)
        matching_shuffled = pd.read_csv(f"../results/{metric}_matching_shuffled.csv", sep='\t', index_col=0)

        base_scores = matching[metric].to_numpy()
        captions_swapped_all = matching_swapped[metric].to_numpy()
        captions_shuffled_all = matching_shuffled[metric].to_numpy()
        n = len(base_scores)
        indices = np.arange(n)
        np.random.shuffle(indices)

        base_score = np.mean(base_scores)
        # Captions from other images 
        print(f"Swapping captions: {metric}")
        for percentage_split in [0, 0.2, 0.5, 0.75, 0.9]:
            score = subsample(captions_swapped_all, base_scores, indices, int(n * percentage_split))
            print(f"{100 * percentage_split}% swapped | {score} | {score / base_score}")
        
        # Shuffled captions
        print(f"Shuffled captions: {metric}")
        for percentage_split in [0, 0.2, 0.5, 0.75, 0.9]:
            score = subsample(captions_shuffled_all, base_scores, indices, int(n * percentage_split))
            print(f"{100 * percentage_split}% shuffled | {score} | {score / base_score}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    flags = parser.parse_args()
  
    main(flags)
