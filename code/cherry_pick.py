from dataset import FlickrDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

dataset = FlickrDataset("../data", type="matching")
base_filepath = "../cherry_pick"
os.makedirs(base_filepath, exist_ok=True)

# for i in range(len(dataset)):
for i in range(100):
    img_filepath, img, caption_id, generated_imgs = dataset[i]
    caption = dataset.annotations[caption_id]
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    fig.suptitle(caption)
    axs[0].imshow(img)
    axs[0].set_title("Flickr Image")
    for j in range(1,4):
        axs[j].imshow(generated_imgs[j-1])
        axs[j].set_title(f"Generated Image {j}")
    fig.tight_layout()
    fig.savefig(os.path.join(base_filepath, f"{i}"))
    plt.close()