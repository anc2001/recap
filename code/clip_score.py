import clip
import torch
from torchvision import transforms

class CLIP_image_score():
    def __init__(self, type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(type, device = self.device)
        model.eval()

        self.input_resolution = model.visual.input_resolution
        self.context_length = model.context_length
        self.vocab_size = model.vocab_size
        self.model = model
        self.preprocess = preprocess
        self.tensor_to_PIL = transforms.ToPILImage()
    
    def __call__(self, real_images, fake_images):
        with torch.no_grad():
            real_image_features = self.model.encode_image(real_images.to(self.device)).float()
            fake_image_features = self.model.encode_image(fake_images.to(self.device)).float()
        real_image_features /= real_image_features.norm(dim=-1, keepdim=True)
        fake_image_features /= fake_image_features.norm(dim=-1, keepdim=True)
        
        return torch.einsum('bd,bd->b', real_image_features, fake_image_features).cpu()