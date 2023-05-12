import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from PIL import Image
import copy
import random
import clip
import pandas as pd
from matplotlib import pyplot as plt

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

        
        




class FlickrDataset(Dataset):

  def __init__(self, train_image_filenames, captions, normalize_prefix=False):

    self.BASE_PATH = './data/Flickr8k/Flicker8k_Dataset'

    self.image_filenames = train_image_filenames
    self.captions = captions

    self.device = torch.device('cuda:6')
    
    # clip.available_models()
    self.clip_model, self.image_preprocessor = clip.load('ViT-L/14')
    self.clip_model.to(self.device)
   

    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    #store max sequence length for padding
    self.max_seq_len = float('-inf')

    

    self.normalize_prefix = normalize_prefix

    for c in self.captions:
        self.max_seq_len = max(len(c.split(' ')), self.max_seq_len)
    
    self.max_seq_len = round(self.max_seq_len, -1)
  
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


    if random.random() <= 1.0:
        shuffled_captions = copy.deepcopy(self.captions[index])

        shuffled_captions = ' '.join(random.sample( shuffled_captions.split(' '), len(shuffled_captions.split(' ')) )) 
    else:
        print('Other mod')
        diff_idx = index

        min_idx = index - (index%5)
        max_idx = index + (5-(index%5))

        invalid_range = range(min_idx, max_idx)

        while diff_idx in invalid_range:
            diff_idx = random.sample(range(len(self.captions)),k=1)[0]

        shuffled_captions = self.captions[diff_idx]

    tokenized_shuffled_captions = self.process_caption(shuffled_captions)

    if self.normalize_prefix:
      image_embedding = image_embedding.float()
      image_embedding = image_embedding / image_embedding.norm(2, -1)
    
    
    # image_embedding = torch.squeeze(image_embedding)
    
    

    return self.image_filenames[index], image_embedding, tokenized_captions, tokenized_shuffled_captions



def train(dataloader, clipcap_model, cross_attention, lr: float = 3e-4, warmup_steps: int = 5000, triplet_margin: float=1.5, triplet_p: int=2, triplet_weight: float=10.0, reconstruction_weight: float=1.0, output_dir: str = "./cross_attention_checkpoints", epochs: int = 10, device=torch.device('cuda:6')):
    # NOTE: in best triplet_weight was 1e-14 and no alternate caption (including with %5 sampling), just shuffled, dropout was 0.25

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # do not train clipcap model nor gpt2 model further
    clipcap_model.to(device)
    clipcap_model.eval()

    # train the cross attention layer though
    cross_attention = cross_attention.to(device)
    cross_attention.train()

    optimizer = AdamW(cross_attention.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(dataloader)
    )

    # define the loss functions used
    triplet_loss = torch.nn.TripletMarginLoss(margin= triplet_margin, p= triplet_p)

    mse_loss = torch.nn.MSELoss()

    # track losses for cross attention layer
    losses_df = pd.DataFrame(columns=['reconstruction_loss','triplet_loss','total_loss'])
    
    for epoch in range(epochs):
        print('Epoch {}/{}:'.format(epoch+1, epochs))

        progress = tqdm(total=len(dataloader))

        for idx, (image_file, image_embedding, tokenized_captions, tokenized_shuffled_captions) in enumerate(dataloader):
            
            
            
            cross_attention.zero_grad()

            image_embedding, tokenized_captions, tokenized_shuffled_captions = image_embedding.to(device), tokenized_captions.to(device), tokenized_shuffled_captions.to(device)
            
            # run non-shuffled tokenized captions and image through clipcap + cross-attention layer
            prefix_projections, embedded_captions = clipcap_model(tokenized_captions, image_embedding)

            prefix_projections, embedded_captions = prefix_projections.to(device), embedded_captions.to(device)
            
            reconstructed_image_embedding = cross_attention(embedded_captions)


            # run shuffled tokenized captions and image through clipcap + cross-attention layer
            shuffled_prefix_projections, shuffled_embedded_captions = clipcap_model(tokenized_shuffled_captions, image_embedding)

            reconstructed_image_embedding_shuffled = cross_attention( shuffled_embedded_captions)
            


            # compute the mse loss on reconstructions against image embedding
            reconstruction_loss = mse_loss(reconstructed_image_embedding, prefix_projections)

            # compute the triplet loss on reconstruction vs shuffled reconstructions
            comparative_loss = triplet_loss(torch.sigmoid(prefix_projections), torch.sigmoid(reconstructed_image_embedding), torch.sigmoid(reconstructed_image_embedding_shuffled))

            total_loss = reconstruction_weight*reconstruction_loss + triplet_weight*comparative_loss


            total_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress.set_postfix({"total loss": total_loss.item(), "reconstruction loss ": reconstruction_weight*reconstruction_loss.item(), "triplet loss": triplet_weight*comparative_loss.item()})
            progress.update()
            
            # append new batch losses to the array
            losses_df.loc[len(losses_df.index)] = [reconstruction_loss.item(), comparative_loss.item(), total_loss.item()]


            if (idx+1)%10000 == 0:
                torch.save( cross_attention.state_dict(),
                os.path.join(output_dir, f"latest_cross_attention.pt"))
        


        progress.close()

        torch.save(cross_attention.state_dict(),
                os.path.join(output_dir, f"cross_attention_epoch{epoch+1}.pt"))
        
        losses_df.to_csv(os.path.join(output_dir, 'losses.csv'), index=False)

        plot_losses(losses_df['total_loss'].values, './cross_attention_checkpoints/train_loss.png', 'Train Loss: caption reconstruction (with cross attention)')
        plot_losses(losses_df['triplet_loss'].values, './cross_attention_checkpoints/triplet_loss.png', 'Triplet Loss: caption reconstruction')
        plot_losses(losses_df['reconstruction_loss'].values, './cross_attention_checkpoints/reconstruction_loss.png', 'Reconstruction Loss: caption reconstruction')
    
    return cross_attention



def plot_losses(train_losses,file_name,title):
    batches = list(map(lambda x: x+1, range(len(train_losses))))

    plt.plot(batches, train_losses, c='b', label = 'train loss')
    plt.legend()
    plt.xlabel('batch number')
    plt.ylabel('train loss (weighted MSE and triplet)')
    plt.title(title)
    plt.savefig(file_name)
    plt.cla()
    plt.clf()

def main():

    # TODO: collect only train image filenames
    f = open(os.path.relpath('./data/Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'), 'r')

    train_image_filenames = []
    train_image_set = []

    for line in tqdm(f.readlines()):
    
        train_image_filenames+= [line[:-1]]*5
        train_image_set.append(line[:-1])
    
     

    
    
    # TODO: create caption dict
    f = open(os.path.relpath('./data/Flickr8k/Flickr8k_text/Flickr8k.token.txt'), 'r')

    caption_dictionary = {}

    for line in tqdm(f.readlines()):

        if line[:line.index('#')] not in caption_dictionary:
            caption_dictionary[line[:line.index('#')]] = [line[line.index('#')+3:].strip().lower()]
        else:
            caption_dictionary[line[:line.index('#')]].append(line[line.index('#')+3:].strip().lower())
    
    train_captions = []

    for image in train_image_set:
        train_captions += caption_dictionary[image]
    
    
    
    # TODO: add in other caption-image pairs with high expert scores
    flickr8k_dataset = FlickrDataset(train_image_filenames, train_captions)
    flickr8k_dataloader = DataLoader(flickr8k_dataset, batch_size=100, shuffle=True)


    # TODO: load all the relevant models
    clipcap_model = ClipCaptionPrefix()
    clipcap_model.load_state_dict(torch.load('./checkpoints/transformer_mapper-009.pt'))
    self_attention = SelfAttentionModule()
    import pdb
    pdb.set_trace()
    #TODO: write training function
    train(flickr8k_dataloader, clipcap_model, self_attention)

if __name__=='__main__':
    
    main()























# class ClipCaptionModel(nn.Module):

#     def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
#         return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

#     def forward(self, tokenized_captions: torch.Tensor, clip_image_embeddings: torch.Tensor):

#         # NOTE: add this to the main training loop since tokenizing has to be done sentence by sentence
#         # tokenized_captions = torch.tensor(self.tokenizer.encode(captions), dtype=torch.int64)
#         pdb.set_trace()
#         embedded_captions = self.gpt.transformer.wte(tokenized_captions)
        

#         prefix_projections = self.clip_project(clip_image_embeddings).view(-1, self.prefix_length, self.gpt_embedding_size)
        
        
#         return prefix_projections, embedded_captions

#     def __init__(self, prefix_length: int = 10, clip_length: Optional[int] = 10, prefix_size: int = 768, num_layers: int = 8):
        
#         super(ClipCaptionModel, self).__init__()

#         self.prefix_length = prefix_length
#         self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
#         self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        
#         self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,clip_length, num_layers)
