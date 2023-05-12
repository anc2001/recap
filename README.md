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

# Codebase Structure

This codebase is divided into 2 main components. The `code` folder contains all the scripts relevant to run baseline evaluations, img-image comparison metrics, stable diffusion, and the latent reconstruction method. The `data` folder (empty in this repo) contains the datasets (Flickr8k) used for running the code

Each file and subfolder serves the following purposes:

- baseline_evaluation: contains notebook scripts used to run the text-text + text-image baseline metrics (BLEU, CIDer, SPICE, METEOR, CLIPScore) against the Flickr8k test set for different degrees of pathological transformations as well as to generate scores for these text-text + text-image metrics for the `ExpertAnnotations.txt` file in the Flickr8k dataset. User `CaptioningBaselines.ipynb` for baseline metrics on test set and `HumanPreferenceBaseline.ipynb` for human preference evaluation

- latent_method: contains python scripts required to load and run the latent reconstruction metrics proposed. Parts of this codebase (for data loading and model setup) were taken from `CLIP_Prefix_caption` i.e. the codebase for ClipCap. After downloading the COCO2015 dataset for captioning and saving it within `data\coco`, use parse `parse_coco.py` and specifying the CLIP image encoder to generate/save image embeddings for training images in the COCO dataset. Also download the `Flickr8k` dataset and save it under `data\Flickr8k`. Use `pretrain_clipcap.py` to pretrain the ClipCap model (preset to use `ViTL-14` image encoder) and save batch-wise loss under `checkpoints`. Then use `train_caption_evaluator_direct.py` (specifying the pretrained model checkpoint you want to use) to further train the latent reconstruction model: keeping GPT-2 encoder and ClipCap base frozen; this script to save the checkpoints of the model as well as figures for per-batch loss in a folder called `cross_attention_checkpoints`. Finally use `evaluate_caption_metric.py` and paste in the appropriate evaluation you want to run within the `name==main` clause to either run inference on pathological transformations, the `ExpertAnnotations.txt` data subset (for human preference correlation), or to visualize the reconstructed embeddings for a batch of 50 example images in the Flickr8k test set.

- Use `dataset.py` provides a pytorch dataset of the Flickr8k image filepaths, corresponding captions, and human evaluations. See class for dataset options.

- Use `stablediff.py` to generate images from captions. By default it generates 3 images per caption.

- Use `img2img.py` for image-image comparison metrics using the generated images from `stablediff.py.`

- `clip_score` is an image-image metric which compares the image embeddings for two different images.

- `humancorr.py` runs a Kendall's Tau correlation for the image-image metrics in `img2img.py` and human annotations present in the Flickr8k dataset.

- `cherry_pick.py` prints image, caption, and generated images.
