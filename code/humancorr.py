import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau 
from img2img import datasets, metrics

os.makedirs("../figures", exist_ok=True)

baseline_metrics = ['bleu1_score', 'bleu2_score', 'bleu3_score', 'bleu4_score', 
                    'meteor_score', 'cider_score', 'spice_score', 'clip_score']

# from https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
def calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = round(kendalltau(tmp[r], tmp[c])[1], 4)
    return pvalues

def plot_correlation(results, metric, dataset, noise=False):
    correlations = results.corr(method="kendall", numeric_only=True)
    pvalues = calculate_pvalues(results[['human_scores', metric]])

    # get data and calculate line of best fit
    x = results.groupby(by=['img_id', 'caption_id'])['human_scores']
    y = results.groupby(by=['img_id', 'caption_id'])[metric]

    if flags.reduce_method == 'mean':
        x, y = x.mean(), y.mean()
    elif flags.reduce_method == 'max':
        x, y = x.max(), y.max()
    elif flags.reduce_method == 'min':
        x, y = x.min(), y.min()

    r = correlations['human_scores'].loc[metric].round(4)
    p = pvalues['human_scores'].loc[metric]

    x, y = results['human_scores'], results[metric]
    slope, intercept = np.polyfit(x, y, 1)
    line = slope * x + intercept
    plt.scatter(x, y, alpha=0.2)
    plt.plot(x, line, color='red')
    plt.title(f"{metric} Correlation with Human Eval " + 
              ("(Noised) " if noise else "") + f"(r={r}, p={p})")
    plt.xlabel("Human Evaluation")
    plt.ylabel(f"{metric}")
    
    if noise:
        save_name = f"../figures/{metric}_{dataset}_{flags.reduce_method}_noise_corr"
    else: 
        save_name = f"../figures/{metric}_{dataset}_{flags.reduce_method}_corr"
    plt.savefig(save_name + ".png")
    plt.clf()

    with open(save_name + ".txt", 'w') as f:
        f.write(f"(r={r}, p={p})")


def main(flags):
    for dataset in datasets.keys():
        for metric in metrics.keys():
            # read in results for metric on given dataset
            try: 
                results = pd.read_csv(f"../results/{metric}_{dataset}.csv", sep='\t', index_col=0)
            except FileNotFoundError:
                continue
            plot_correlation(results, metric, dataset)
            # TODO how do we easily visually show the baseline inter-image metrics?
            # histogram of differences between the two?? not sure. 

            # do the same thing for metric comparing ground truth to noise
            try: 
                noise = pd.read_csv(f"../results/{metric}_{dataset}_noise.csv", sep='\t', index_col=0)
            except FileNotFoundError:
                continue
            plot_correlation(noise, metric, dataset, noise=True)
            # TODO how do we easily visually show the baseline inter-image metrics?
            # histogram of differences between the two?? not sure. 


    # get scatterplots for baseline as well
    baselines_joined = pd.read_csv("../data/baselines_joined.csv", index_col=0)
    for metric in baseline_metrics:
        plot_correlation(baselines_joined, metric, "annotated") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduce_method", choices=['mean', 'max', 'min'], default='mean')
    parser.set_defaults(shuffled=False)
    flags = parser.parse_args()
  
    main(flags)


