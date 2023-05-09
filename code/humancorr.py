import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from img2img import datasets, metrics

os.makedirs("../figures", exist_ok=True)

for dataset in datasets.keys():
    for metric in metrics.keys():
        try:
            results = pd.read_csv(f"../results/{metric}_{dataset}.csv", sep='\t', index_col=0)
        except FileNotFoundError:
            continue
        correlations = results.corr(numeric_only=True)

        # calculate line of best fit 
        x, y = results['human_scores'], results[metric]
        slope, intercept = np.polyfit(x, y, 1)
        line = slope * x + intercept
        plt.scatter(x, y)
        plt.plot(x, line, color='red')
        plt.text(0.89, 0.95, f"r={correlations['human_scores'].loc[metric].round(4)}")
        plt.title(f"Correlation of {metric} with Human Eval for Swapped+Matching Captions")
        plt.xlabel("Human Evaluation")
        plt.ylabel(f"{metric}")
        plt.savefig(f"../figures/{metric}_{dataset}_corr.png")
        plt.clf()

        # TODO you have to actulaly do averaging over the three generated images
        # results.groupby(by=['img_id', 'caption_id'])['human_scores'].mean()
        # results.groupby(by=['img_id', 'caption_id'])[metric].mean()

        # TODO also try doing it by max and min over all three generated images 
        # results.groupby(by=['img_id', 'caption_id'])['human_scores'].max()
        # results.groupby(by=['img_id', 'caption_id'])[metric].max()

        # TODO how do we easily visually show the baseline inter-image metrics?
        # histogram of differences between the two?? not sure. 




