import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
from torch.nn import functional as nnf
import pandas as pd
import numpy as np
import random
import copy
from matplotlib import pyplot as plt
import pdb

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)



class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        
        
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length*self.gpt_embedding_size)
       
        return prefix_projections, embedding_text

    def __init__(self, prefix_length: int=10, clip_length: int = 10, prefix_size: int = 768, num_layers: int = 8):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
       
        self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length, clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


class SelfAttentionModule(nn.Module):
    def forward(self, embedded_caption):
        
        attn_output, _ = self.cross_attention_layer(embedded_caption, embedded_caption, embedded_caption)

        attn_output = attn_output.reshape(-1, attn_output.shape[1]*attn_output.shape[2])

        output = self.inverse_projection(attn_output)

        
        return output

    def compute_metric(self, reconstruction_losses):
        return torch.mean(reconstruction_losses, dim=1), torch.sqrt(torch.sum(reconstruction_losses, dim=1))

    def __init__(self, num_heads: int = 12, clip_image_embedding_length: int = 768, max_caption_length:int=40, prefix_length: int = 10, gpt2_embedding_length: int = 768):

        # NOTE: embed_dim should = clip_image_embedding_length
        # NOTE: kdim & vdim should both = SAME thing (clip_image_embedding_length or gpt2_embedding_length)

        # NOTE: prefix_dim = clip_image_embedding_length
        
        """
        image embeddings as the "key" and "value" tensors and the language embeddings as the "query" tensor
        """

        super(SelfAttentionModule, self).__init__()

        self.cross_attention_layer = torch.nn.MultiheadAttention(embed_dim = clip_image_embedding_length, num_heads = num_heads, dropout=0.25, kdim=gpt2_embedding_length, vdim=gpt2_embedding_length, batch_first=True)

        self.inverse_projection = MLP((gpt2_embedding_length*max_caption_length,(gpt2_embedding_length*max_caption_length)//2, clip_image_embedding_length*prefix_length))

        self.mse_loss = torch.nn.MSELoss(reduction='none')

      


class FlickrDatasetPathological(Dataset):

  def __init__(self, test_image_filenames, caption_dict, modified, normalize_prefix=False):

    self.BASE_PATH = './data/Flickr8k/Flicker8k_Dataset'

    self.image_filenames = test_image_filenames
    self.captions = caption_dict

    self.device = torch.device('cuda:6')
    
    # clip.available_models()
    self.clip_model, self.image_preprocessor = clip.load('ViT-L/14')
    self.clip_model.to(self.device)
   

    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    #store max sequence length for padding
    self.max_seq_len = 40

    #store if each of the images are modified or not (for pathological captions)
    self.modified = modified

    self.normalize_prefix = normalize_prefix

    
  
  def image_load_preprocess_embed(self, filename):

    image = Image.open(os.path.join(self.BASE_PATH, filename)).convert("RGB")
    
    image = self.image_preprocessor(image)
    
    
    image = torch.unsqueeze(image, dim=0)
    image = image.to(self.device)
    

    
    with torch.no_grad():
      image_embedding = self.clip_model.encode_image(image).float()
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

    return image_embedding
  
  def process_caption(self, caption):
    
    tokenized_caption = self.tokenizer.encode(caption)
    tokenized_caption = torch.tensor(tokenized_caption, dtype=torch.int64)
    
    
    # pad the token to max length if required
    padding = self.max_seq_len - tokenized_caption.shape[0]
      
    if padding >= 0:
        tokenized_caption = torch.cat((tokenized_caption, torch.zeros(padding, dtype=torch.int64) - 1))
    else:
        tokenized_caption = tokenized_caption[:self.max_seq_len]
    
    mask = tokenized_caption.ge(0)  
    tokenized_caption[~mask] = 0
            
    return tokenized_caption
  
  def __len__(self):
    return len(self.image_filenames)
  
  def __getitem__(self, index):
   
    
    image_embedding = self.image_load_preprocess_embed(self.image_filenames[index])
    
    
    tokenized_captions = self.process_caption(self.captions[index])


    if self.normalize_prefix:
      image_embedding = image_embedding.float()
      image_embedding = image_embedding / image_embedding.norm(2, -1)
    
    
    # image_embedding = torch.squeeze(image_embedding)
    
    return self.image_filenames[index], self.captions[index], image_embedding, tokenized_captions, self.modified[index]




class FlickrDatasetPreference(Dataset):

  def __init__(self, human_preference_df, caption_dict, normalize_prefix=False):

    self.BASE_PATH = './data/Flickr8k/Flicker8k_Dataset'

    self.human_preference_df = human_preference_df
    self.caption_dict = caption_dict

    self.device = torch.device('cuda:6')
    
    # clip.available_models()
    self.clip_model, self.image_preprocessor = clip.load('ViT-L/14')
    self.clip_model.to(self.device)
   

    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    #store max sequence length for padding
    self.max_seq_len = 40

    self.normalize_prefix = normalize_prefix

    
  
  def image_load_preprocess_embed(self, filename):

    image = Image.open(os.path.join(self.BASE_PATH, filename)).convert("RGB")
    
    image = self.image_preprocessor(image)
    
    
    image = torch.unsqueeze(image, dim=0)
    image = image.to(self.device)
    

    
    with torch.no_grad():
      image_embedding = self.clip_model.encode_image(image).float()
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

    return image_embedding
  
  def process_caption(self, caption):
    
    tokenized_caption = self.tokenizer.encode(caption)
    tokenized_caption = torch.tensor(tokenized_caption, dtype=torch.int64)
    
    
    # pad the token to max length if required
    padding = self.max_seq_len - tokenized_caption.shape[0]
      
    if padding >= 0:
        tokenized_caption = torch.cat((tokenized_caption, torch.zeros(padding, dtype=torch.int64) - 1))
    else:
        tokenized_caption = tokenized_caption[:self.max_seq_len]
    
    mask = tokenized_caption.ge(0)  
    tokenized_caption[~mask] = 0
            
    return tokenized_caption
  
  def __len__(self):
    return len(self.human_preference_df)
  
  def __getitem__(self, index):
   
    row = self.human_preference_df.iloc[index]

    image_embedding = self.image_load_preprocess_embed(row['image_filename'])
    
    
    tokenized_captions = self.process_caption(self.caption_dict[row['target_caption']][2])


    if self.normalize_prefix:
      image_embedding = image_embedding.float()
      image_embedding = image_embedding / image_embedding.norm(2, -1)
    
    
    # image_embedding = torch.squeeze(image_embedding)
    return row['image_filename'], self.caption_dict[row['target_caption']][2], image_embedding, tokenized_captions



def compute_all_latent_reconstruction_scores(image_dataloader, clipcap_model, self_attention, experiment='test'):

    avg_mean_latent_score = 0 
    avg_sum_latent_score = 0 

    latent_df = pd.DataFrame(columns=['image_file','ref','latent_score_mean','latent_score_sum', 'modified'])

    device = torch.device('cuda:6')

    clipcap_model.to(device)
    clipcap_model.eval()

    self_attention.to(device)
    self_attention.eval()

    for idx, (image_filenames, captions, image_embeddings, tokenized_captions, modified) in enumerate(image_dataloader):
    
        print('Processing image {} of {}'.format(idx+1, len(image_dataloader)))

        image_embeddings, tokenized_captions = image_embeddings.to(device), tokenized_captions.to(device)

        with torch.no_grad():
            # first pass caption and image embedding through pretrained clipcap model and GPT2 encoder
            prefix_projections, embedded_captions = clipcap_model(tokenized_captions, image_embeddings)

            prefix_projections, embedded_captions = prefix_projections.to(device), embedded_captions.to(device)
            
            #reconstruct the image embedding in text space i.e. prefix projection
            reconstructed_image_embeddings = self_attention(embedded_captions)

            #compute reconstruction MSE loss as evaluation metric
            reconstruction_losses = self_attention.mse_loss(reconstructed_image_embeddings, prefix_projections)
        
            mean_losses, sum_losses = self_attention.compute_metric(reconstruction_losses)

        mean_losses = mean_losses.cpu().numpy()
        sum_losses = sum_losses.cpu().numpy()

        

        batch_df = pd.DataFrame.from_dict({'image_file': image_filenames, 'ref': captions, 'latent_score_mean': mean_losses, 'latent_score_sum': sum_losses, 'modified': modified.numpy()},orient='index').transpose()

        latent_df = latent_df.append(batch_df, ignore_index=True)

        avg_mean_latent_score += np.sum(mean_losses)
        avg_sum_latent_score += np.sum(sum_losses)
    
    latent_df.to_csv('./results/latent_scores_{}.csv'.format(experiment), index=False)
    return avg_mean_latent_score/(1000*5), avg_sum_latent_score/(1000*5)
            


def compute_all_latent_reconstruction_scores_pca(image_dataloader, clipcap_model, self_attention, experiment='test'):


    device = torch.device('cuda:6')

    clipcap_model.to(device)
    clipcap_model.eval()

    self_attention.to(device)
    self_attention.eval()

    for idx, (image_filenames, captions, image_embeddings, tokenized_captions, modified) in enumerate(image_dataloader):
    
        print('Processing image {} of {}'.format(idx+1, len(image_dataloader)))

        image_embeddings, tokenized_captions = image_embeddings.to(device), tokenized_captions.to(device)

        with torch.no_grad():
            # first pass caption and image embedding through pretrained clipcap model and GPT2 encoder
            prefix_projections, embedded_captions = clipcap_model(tokenized_captions, image_embeddings)

            prefix_projections, embedded_captions = prefix_projections.to(device), embedded_captions.to(device)
            
            #reconstruct the image embedding in text space i.e. prefix projection
            reconstructed_image_embeddings = self_attention(embedded_captions)

        
            reduced_reconstructed_embeddings,_,_ = torch.pca_lowrank(reconstructed_image_embeddings, center=True, q=30, niter=15)


            reduced_gt_embeddings,_,_ = torch.pca_lowrank(prefix_projections, center=True, q=30, niter=15)

            
            temp1 = self_attention.mse_loss(reconstructed_image_embeddings[:10,:], prefix_projections[:10,:])

            mean_losses, sum_losses = self_attention.compute_metric(temp1)
            print(mean_losses)

            temp2 = self_attention.mse_loss(reconstructed_image_embeddings[50:60,:], prefix_projections[50:60,:])

            mean_losses, sum_losses = self_attention.compute_metric(temp2)
            print(mean_losses)

            print(captions[:10])
            print(captions[50:60])

            plt.matshow(reduced_reconstructed_embeddings[:10,:].cpu().numpy()/np.max(reduced_reconstructed_embeddings[:10,:].cpu().numpy()), cmap='viridis')
            
            plt.savefig('./normal_embedding.png')
            plt.cla()
            plt.clf()


            plt.matshow(reduced_reconstructed_embeddings[40:50,:].cpu().numpy()/np.max(reduced_reconstructed_embeddings[40:50,:].cpu().numpy()), cmap='viridis')
            
            plt.savefig('./other_embedding.png')
            plt.cla()
            plt.clf()


            plt.matshow(reduced_reconstructed_embeddings[50:60,:].cpu().numpy()/np.max(reduced_reconstructed_embeddings[50:60,:].cpu().numpy()), cmap='viridis')
            
            plt.savefig('./shuffled_embedding.png')
            plt.cla()
            plt.clf()


            

            plt.matshow(reduced_gt_embeddings[:10,:].cpu().numpy()/np.max(reduced_gt_embeddings[:10,:].cpu().numpy()), cmap='viridis')
            plt.savefig('./gt_embedding.png')
            plt.cla()
            plt.clf()

            pdb.set_trace()
        


def compute_all_latent_reconstruction_scores_human(image_dataloader, clipcap_model, self_attention):

    all_mean_latent_scores= []
    all_sum_latent_scores = []

    latent_df = pd.read_csv('./results/human_preference.csv')

    device = torch.device('cuda:6')

    clipcap_model.to(device)
    clipcap_model.eval()

    self_attention.to(device)
    self_attention.eval()

    for idx, (image_filenames, captions, image_embeddings, tokenized_captions) in enumerate(image_dataloader):
    
        print('Processing image {} of {}'.format(idx+1, len(image_dataloader)))

        image_embeddings, tokenized_captions = image_embeddings.to(device), tokenized_captions.to(device)

        with torch.no_grad():
            # first pass caption and image embedding through pretrained clipcap model and GPT2 encoder
            prefix_projections, embedded_captions = clipcap_model(tokenized_captions, image_embeddings)

            prefix_projections, embedded_captions = prefix_projections.to(device), embedded_captions.to(device)
            
            #reconstruct the image embedding in text space i.e. prefix projection
            reconstructed_image_embeddings = self_attention(embedded_captions)

            #compute reconstruction MSE loss as evaluation metric
            reconstruction_losses = self_attention.mse_loss(reconstructed_image_embeddings, prefix_projections)
        
            mean_losses, sum_losses = self_attention.compute_metric(reconstruction_losses)

        mean_losses = list(mean_losses.cpu().numpy())
        sum_losses = list(sum_losses.cpu().numpy())

        all_mean_latent_scores += mean_losses
        all_sum_latent_scores += sum_losses

    
    
    pdb.set_trace()

    latent_df['mean_score'] = all_mean_latent_scores
    latent_df['sum_score'] = all_sum_latent_scores
    
    latent_df.to_csv('./results/human_preferences.csv', index=False)
    
            


    


def run_pathological_captions_test():


    #load clipcap model from pretrained weights
    clipcap_model = ClipCaptionPrefix()
    clipcap_model.load_state_dict(torch.load('./checkpoints/transformer_mapper-009.pt'))

    # load selfattention module from pretrained weights
    self_attention = SelfAttentionModule()
    self_attention.load_state_dict(torch.load('./cross_attention_checkpoints/cross_attention_epoch6.pt'))

    
    if not os.path.exists('./results'):
        os.mkdir('./results')

    f = open(os.path.relpath('./data/Flickr8k/Flickr8k_text/Flickr_8k.testImages.txt'), 'r')

    test_image_set = []
    test_image_filenames = []

    for line in tqdm(f.readlines()):
        test_image_filenames += [line[:-1]]*5
        test_image_set.append(line[:-1])


    print('Test Set Length: {} \n'.format(len(test_image_filenames)))


    f = open(os.path.relpath('./data/Flickr8k/Flickr8k_text/Flickr8k.token.txt'), 'r')

    caption_dictionary = {}

    for line in tqdm(f.readlines()):
        
        if line[:line.index('#')] not in test_image_set:
            continue

        if line[:line.index('#')] not in caption_dictionary:
            caption_dictionary[line[:line.index('#')]] = [line[line.index('#')+3:].strip().lower()]
        else:
            caption_dictionary[line[:line.index('#')]].append(line[line.index('#')+3:].strip().lower())

    
    shuffle = False
    percent_shuffled_captions = 0.90

    swap = False
    percent_swapped_captions = 0.90

    test_pca = True

    if shuffle:

        examples_to_skew = random.sample(list(caption_dictionary.keys()), k=int(len(caption_dictionary.keys())*percent_shuffled_captions))

        caption_dict_shuffled = copy.deepcopy(caption_dictionary)

        for f in examples_to_skew:
            caption_dict_shuffled[f] = [ ' '.join(random.sample( c.split(' '), len(c.split(' ')) )) for c in caption_dict_shuffled[f]]
        
        test_captions = []
        modified = []

        for image in test_image_set:
            test_captions += caption_dict_shuffled[image]
            if image in examples_to_skew:
                modified += [1]*5
            else:
                modified += [0]*5

    elif swap:

        examples_to_skew = random.sample(list(caption_dictionary.keys()), k=int(len(caption_dictionary.keys())*percent_swapped_captions))

        swapped_captions = []
        for e in examples_to_skew:
            swapped_captions += caption_dictionary[e]

        random.shuffle(swapped_captions)
        swapped_captions = [swapped_captions[n:n+5] for n in range(0, len(swapped_captions), 5)]

        #make copy of caption dictionary
        caption_dict_swapped = copy.deepcopy(caption_dictionary)


        for i, e in enumerate(examples_to_skew):
        
            caption_dict_swapped[e] = swapped_captions[i]
        
        test_captions = []
        modified = []

        for image in test_image_set:
            test_captions += caption_dict_swapped[image]
            if image in examples_to_skew:
                modified += [1]*5
            else:
                modified += [0]*5

    else:

        modified = []
        test_captions = []
        for image in test_image_set:
            test_captions += caption_dictionary[image]
            modified += [0]*5
    
    if test_pca:
        pca_examples = random.sample(list(caption_dictionary.keys()), k=50)

        caption_dict_shuffled = copy.deepcopy(caption_dictionary)

        for f in pca_examples:
            caption_dict_shuffled[f] = [ ' '.join(random.sample( c.split(' '), len(c.split(' ')) )) for c in caption_dict_shuffled[f]]
        
        normal_test_captions = []
        modified_normal = []

        for image in pca_examples:
            normal_test_captions.append(caption_dictionary[image][0])
            modified_normal += [0]
        
        modified_test_captions = []
        modified_modified = []

        for image in pca_examples:
            modified_test_captions.append(caption_dict_shuffled[image][0])
            modified_modified += [0]
        

        pca_test_images = pca_examples + pca_examples
        pca_test_captions = normal_test_captions + modified_test_captions

        modified = modified_normal + modified_modified
        
        dataset = FlickrDatasetPathological(pca_test_images, pca_test_captions, modified)
        image_dataloader = torch.utils.data.DataLoader(dataset, batch_size=100)

        compute_all_latent_reconstruction_scores_pca(image_dataloader, clipcap_model, self_attention, experiment='flickr8k_pca')
        


        



    print('Num Modified: ', np.sum(modified))
    import pdb
    pdb.set_trace()

    dataset = FlickrDatasetPathological(test_image_filenames, test_captions, modified)
    image_dataloader = torch.utils.data.DataLoader(dataset, batch_size=50)

    
    if shuffle and not test_pca:
        avg_mean_score, avg_sum_score = compute_all_latent_reconstruction_scores(image_dataloader, clipcap_model, self_attention, experiment='flickr8k_shuffled_{}'.format(int(percent_shuffled_captions*100)))
    elif swap and not test_pca:
        avg_mean_score, avg_sum_score = compute_all_latent_reconstruction_scores(image_dataloader, clipcap_model, self_attention, experiment='flickr8k_other_images_{}'.format(int(percent_shuffled_captions*100)))
    elif not test_pca:
        avg_mean_score, avg_sum_score = compute_all_latent_reconstruction_scores(image_dataloader, clipcap_model, self_attention, experiment='flickr8k')

    print('Average Mean Score: ', avg_mean_score)
    print('Average Sum Score: ', avg_sum_score)


def run_human_preferences_test():

    #load clipcap model from pretrained weights
    clipcap_model = ClipCaptionPrefix()
    clipcap_model.load_state_dict(torch.load('./checkpoints/transformer_mapper-009.pt'))

    # load selfattention module from pretrained weights
    self_attention = SelfAttentionModule()
    self_attention.load_state_dict(torch.load('./cross_attention_checkpoints/cross_attention_epoch6.pt'))


    if not os.path.exists('./results'):
        os.mkdir('./results')


    expert_annotations = set()

    file_exists = os.path.isfile('./results/human_preference.csv')

    if file_exists:
        human_preference_df = pd.read_csv('./results/human_preference.csv')
    else:
        human_preference_df = pd.DataFrame(columns=['image_filename','target_caption','gt_captions'])


    f = open(os.path.relpath('./data/Flickr8k/Flickr8k_text/ExpertAnnotations.txt'), 'r')

    for line in tqdm(f.readlines()):
    
        data_line = line.split('\n')[0].split('\t')
        expert_annotations.add(data_line[0])
        expert_annotations.add(data_line[1].split('#')[0])

        if not file_exists:
            human_preference_df.loc[len(human_preference_df.index)] = [data_line[0], data_line[1].split('#')[0], data_line[0]] 

    if not file_exists:
        human_preference_df.to_csv('./results/human_preference.csv', index=False)

    print('\nExpert Annotations Length: {} \n'.format(len(expert_annotations)))

    f = open(os.path.relpath('./data/Flickr8k/Flickr8k_text/Flickr8k.token.txt'), 'r')

    caption_dictionary = {}

    for line in tqdm(f.readlines()):

        if line[:line.index('#')] not in expert_annotations:
            continue

        if line[:line.index('#')] not in caption_dictionary:
            caption_dictionary[line[:line.index('#')]] = [line[line.index('#')+3:].strip().lower()]
        else:
            caption_dictionary[line[:line.index('#')]].append(line[line.index('#')+3:].strip().lower())
    

    import pdb
    pdb.set_trace()
    dataset = FlickrDatasetPreference(human_preference_df, caption_dictionary)
    image_dataloader = torch.utils.data.DataLoader(dataset, batch_size=50)

    compute_all_latent_reconstruction_scores_human(image_dataloader, clipcap_model, self_attention)


if __name__ == '__main__':
    run_pathological_captions_test()