# Recap

# Environment Install
Create conda environment `recap`
```
conda env create -f environment.yaml
conda activate recap
```

Create conda environment `caption_project` for latent reconstruction method

```
conda env create -f environment_latent.yml
conda activate caption_project
```

Find the `environment_latent.yml` file under `code/latent_method`

#Codebase Structure

This codebase is divided into 2 main components. The `code` folder contains all the scripts relevant to run baseline evaluations, img-image comparison metrics, stable diffusion, and the latent reconstruction method. The `data` folder (empty in this repo) contains the datasets (Flickr8k) used for running the code

Each file and subfolder serves the following purposes:

- baseline_evaluation: contains notebook scripts used to run the text-text + text-image baseline metrics (BLEU, CIDer, SPICE, METEOR, CLIPScore) against the Flickr8k test set for different degrees of pathological transformations as well as to generate scores for these text-text + text-image metrics for the `ExpertAnnotations.txt` file in the Flickr8k dataset. User `CaptioningBaselines.ipynb` for baseline metrics on test set and `HumanPreferenceBaseline.ipynb` for human preference evaluation

- latent_method: contains python scripts required to load and run the latent reconstruction metrics proposed. Parts of this codebase (for data loading and model setup) were taken from `CLIP_Prefix_caption` i.e. the codebase for ClipCap. After downloading the COCO2015 dataset for captioning and saving it within `data\coco`, use parse `parse_coco.py` and specifying the CLIP image encoder to generate/save image embeddings for training images in the COCO dataset. Also download the `Flickr8k` dataset and save it under `data\Flickr8k`. Use `pretrain_clipcap.py` to pretrain the ClipCap model (preset to use `ViTL-14` image encoder) and save batch-wise loss under `checkpoints`. Then use `train_caption_evaluator_direct.py` (specifying the pretrained model checkpoint you want to use) to further train the latent reconstruction model: keeping GPT-2 encoder and ClipCap base frozen; this script to save the checkpoints of the model as well as figures for per-batch loss in a folder called `cross_attention_checkpoints`. Finally use `evaluate_caption_metric.py` and paste in the appropriate evaluation you want to run within the `name==main` clause to either run inference on pathological transformations, the `ExpertAnnotations.txt` data subset (for human preference correlation), or to visualize the reconstructed embeddings for a batch of 50 example images in the Flickr8k test set.

- `img2img.py` and `stablediff.py` are taken from StableDiffusion's codebase and is used for explicit image reconstruction going from text prompt to image. Either `evaluate_metric.py` or `clip_score.py` can then be used to evaluate metrics from GAN literature or compare the CLIP image embeddings of the original and reconstructed images. The `humancorr.py` is used to run Kendall's Tau correlation for all proposed caption metrics; it references the `ExpertAnnotations.txt` expert scores (from Flickr8k) and correlates the score assigned by metrics with human evaluations. Finally the `cherry_pick.py` file chooses specific qualitative examples of reconstructed images and saves them.
