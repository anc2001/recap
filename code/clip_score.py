import clip
import torch

class CLIP_image_score():
    def __init__(self, type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(type, device = self.device)
        model.eval()

        self.input_resolution = model.visual.input_resolution
        self.context_length = model.context_length
        self.vocab_size = model.vocab_size
        self.model = model
        self.preprocess =preprocess
    
    def __call__(self, real_images, fake_images):
        real_images = self.preprocess(real_images).to(self.device)
        fake_images = self.preprocess(fake_images).to(self.device)
        with torch.no_grad():
            real_image_features = self.model.encode_image(real_images).float()
            fake_image_features = self.model.encode_image(fake_images).float()
        real_image_features /= real_image_features.norm(dim=-1, keepdim=True)
        fake_image_features /= fake_image_features.norm(dim=-1, keepdim=True)
        
        return torch.dot(real_image_features, fake_image_features)
