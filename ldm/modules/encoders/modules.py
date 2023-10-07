import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
import kornia
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
        
class Encoder4(nn.Module):
    def __init__(self, d, bn=True, num_channels=3, latent_dim=192):
        super(Encoder4, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            View((-1, 128 * 4 * 4)),                  # batch_size x 2048
            nn.Linear(2048, self.latent_dim)
        )
    def forward(self,x):
        return self.encoder(x)

class Encoder4_vae(nn.Module):
    def __init__(self, d, bn=True, num_channels=3, latent_dim=192):
        super(Encoder4_vae, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            View((-1, 128 * 4 * 4)),                  # batch_size x 2048
            nn.Linear(2048, 2*self.latent_dim)
        )
    def forward(self,x):
        moments = self.encoder(x)
        self.posteriors = DiagonalGaussianDistribution(moments)
        return self.posteriors.sample()
    
    def kl_loss(self, latent_unit):
        kl_loss_splits = self.posteriors.kl_splits(latent_unit=latent_unit)
        return kl_loss_splits


class Encoder256(nn.Module):
    def __init__(self, d, bn=True, num_channels=3, latent_dim=192):
        super(Encoder256, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            View((-1, 128 * 4 * 4)),                  # batch_size x 2048
            nn.Linear(2048, self.latent_dim)
        )
    def forward(self,x):
        return self.encoder(x)

from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class MLP_head(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_cls)
        )
    def forward(self, x):
        return self.net(x)

# main class
class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        *,
        index_num = 32,
        depth=4,
        dim=32,
        z_index_dim = 10,
        latent_dim = 32,
        cross_heads = 1,
        latent_heads = 3,
        cross_dim_head = 32,
        latent_dim_head = 32,
        weight_tie_layers = False,
        max_freq = 10,
        num_freq_bands = 6
    ):
        super().__init__()
        self.num_latents = z_index_dim
        self.components = z_index_dim
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.depth = depth

        self.encoder = nn.Sequential(
            nn.Conv2d(3, latent_dim // 2, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            ResBlock(latent_dim, latent_dim, bn=True),
            nn.BatchNorm2d(latent_dim),
            ResBlock(latent_dim, latent_dim, bn=True),
        )


        self.latents = nn.Parameter(torch.randn(self.num_latents, latent_dim), True)
        self.cs_layers = nn.ModuleList([])
        for i in range(depth):
            self.cs_layers.append(nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, dim + 26, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim + 26),
                PreNorm(latent_dim, FeedForward(latent_dim))
            ]))

        get_latent_attn = lambda: PreNorm(dim + 26, Attention(dim + 26, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(dim + 26, FeedForward(dim + 26))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth-1):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))
        self.fc_layer = nn.Linear(dim, index_num)

    def forward(
        self,
        data,
        mask = None
    ):
        data = self.encoder(data)
        data = data.reshape(*data.shape[:2],-1).permute(0,2,1)
        b, *axis, device = *data.shape, data.device
        
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), (int(np.sqrt(axis[0])),int(np.sqrt(axis[0])))))
        pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b = b)

        data = torch.cat((data, enc_pos.reshape(b,-1,enc_pos.shape[-1])), dim = -1)
        x0 = repeat(self.latents, 'n d -> b n d', b = b)
        for i in range(self.depth):
            cross_attn, cross_ff = self.cs_layers[i]

            # cross attention only happens once for Perceiver IO

            x = cross_attn(x0, context = data, mask = mask) + x0
            x0 = cross_ff(x) + x

            if i != self.depth - 1:
                self_attn, self_ff = self.layers[i]
                x_d = self_attn(data) + data
                data = self_ff(x_d) + x_d

        return self.fc_layer(x0).reshape(x0.shape[0],-1)

def swish(x):
    return x * torch.sigmoid(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)




class MLP_layer(nn.Module):
    def __init__(self, z_dim = 512, latent_dim = 256):
        super(MLP_layer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self,x):
        return self.net(x)

class MLP_layers(nn.Module):
    def __init__(self, z_dim = 512, latent_dim = 256, num_latents=16):
        super(MLP_layers, self).__init__()
        self.nets = nn.ModuleList([MLP_layer(z_dim=z_dim, latent_dim=latent_dim) for i in range(num_latents)])


    def forward(self,x):
        out = []
        for sub_net in self.nets:
            out.append(sub_net(x)[:,None,:])
        return torch.cat(out, dim=1)

class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        *,
        depth = 6,
        index_num = 10,
        dim=256,
        z_index_dim = 64,
        latent_dim = 256,
        cross_heads = 1,
        cross_dim_head = 128,
        latent_heads = 6,
        latent_dim_head = 128,
        fourier_encode_data = False,
        weight_tie_layers = False,
        max_freq = 10,
        num_freq_bands = 6,
    ):
        super().__init__()
        num_latents = z_index_dim
        self.components = z_index_dim
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.fourier_encode_data = fourier_encode_data
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim), True)

        self.depth = depth
        if depth != 0:
            get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = latent_heads, dim_head = latent_dim_head))
            get_latent_ff = lambda: PreNorm(dim, FeedForward(dim))
            get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

            self.slayers = nn.ModuleList([])
            cache_args = {'_cache': weight_tie_layers}

            for i in range(depth-1):
                self.slayers.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

        self.cs_layers = nn.ModuleList([])
        for i in range(depth):
            self.cs_layers.append(nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
                PreNorm(latent_dim, FeedForward(latent_dim))
            ]))
        self.fc_layer = nn.Linear(dim, index_num)
        
        if depth != 0:
            get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
            get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
            get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

            self.layers = nn.ModuleList([])
            cache_args = {'_cache': weight_tie_layers}

            for i in range(depth):
                self.layers.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))


    def forward(
        self,
        data,
        mask = None
    ):
        b, *axis, device = *data.shape, data.device
        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), (int(np.sqrt(axis[0])),int(np.sqrt(axis[0])))))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            data = torch.cat((data, enc_pos.reshape(b,-1,enc_pos.shape[-1])), dim = -1)

        x = repeat(self.latents, 'n d -> b n d', b = b)
        cp_vals = data
        for i in range(self.depth):

            cross_attn, cross_ff = self.cs_layers[i]
            x = cross_attn(x, context = cp_vals, mask = mask) + x
            x = cross_ff(x) + x


            self_attn, self_ff = self.layers[i]
            x = self_attn(x) + x
            x = self_ff(x) + x

            if i != self.depth - 1:
                self_attn, self_ff = self.slayers[i]
                cp_vals = self_attn(cp_vals) + cp_vals
                cp_vals = self_ff(cp_vals) + cp_vals

        return self.fc_layer(x)