#defines the base model class this will be using a unet model as defined in the paper
from abc import abstractmethod
import torch as th
from torch.nn.modules.dropout import Dropout
from torchinfo import summary
from torch import nn
import torchvision.models
from torchvision.models.resnet import resnext50_32x4d
import math
import numpy as np
import torch.nn.functional as F

from nn import avg_pool_nd, linear , conv_nd , normalization , zero_module, timestep_embedding

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]



class Downsample(nn.Module):
  def __init__(self , channels , use_conv , dims=2 , out_channels=None):
    super().__init__()
    self.channels = channels
    self.out_channels = out_channels or channels
    self.use_conv = use_conv
    self.dims = dims
    stride = 2 if dims != 3 else (1 , 2 , 2)
    if(use_conv):
      self.op = conv_nd(dims, in_channels=channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
    else:
      assert self.channels == self.out_channels
      self.op = avg_pool_nd(dims , kernel_size=stride, stride=stride)
  
  def forward(self , x):
    assert x.shape[1] == self.channels
    return self.op(x)

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
      
      
class ResBlock(TimestepBlock):
  def __init__(self , channels , emb_channels, dropout=0.5 , out_channels=None , dims=2, use_conv=False, use_scale_shift_norm=False, use_checkpoints=False , up=False , down=False):
    super().__init__()
    self.channels = channels
    self.emb_channels = emb_channels
    self.dropout = dropout
    self.out_channels = out_channels or channels
    self.use_conv = use_conv
    self.up_down = up or down
    self.use_scale_shift_norm = use_scale_shift_norm
    self.in_layers = nn.Sequential(
          normalization(channels),
          nn.SiLU(),
          conv_nd(dims, channels, self.out_channels, 3, padding=1),
      )
    if up:
        self.h_upd = Upsample(channels, False, dims)
        self.x_upd = Upsample(channels, False, dims)
    elif down:
        self.h_upd = Downsample(channels, False, dims)
        self.x_upd = Downsample(channels, False, dims)
    else:
        self.h_upd = self.x_upd = nn.Identity()
    
    self.emb_layers = nn.Sequential(
      nn.SiLU(),
      linear(emb_channels,2 * self.out_channels if use_scale_shift_norm else self.out_channels,),
      )
    
    self.out_layers = nn.Sequential(
      normalization(self.out_channels),
      nn.SiLU(),
      nn.Dropout(self.dropout),
      zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))
      )
    
    if self.out_channels == channels:
      self.skip_connection = nn.Identity()
    elif use_conv:
      self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
    else:
      self.skip_connection =  conv_nd(dims, channels, self.out_channels, 1)

    
    
  def forward(self , x , emb):
    return self._forward(x , emb)
  
  def _forward(self , x , emb):
    if self.up_down:
        in_rest , in_conv = self.in_layers[:-1] , self.in_layers[-1]
        h = in_rest(x)
        h = self.h_upd(h)
        x = self.x_upd(x)
        h = in_conv(h)
    else:
        h = self.in_layers(x)
    emb_out = self.emb_layers(emb).type(h.dtype)
    while(len(emb_out.shape) < len(h.shape)):
      emb_out = emb_out[... , None]
    if self.use_scale_shift_norm:
      out_norm , out_rest = self.out_layers[0] , self.out_layers[1:]
      scale , shift = th.chunk(emb_out , 2 , dim=1)
      h = out_norm(h) * (1 + scale) + shift
      h = out_rest(h)
    else:
      h = h + emb_out
      h = self.out_layers(h)
    return self.skip_connection(x) + h
      
class AttentionBlock(nn.Module):
  
  #the attention block that allows network to learn spatial attention
  
  def __init__(self , channels , num_heads , num_head_channels , use_new_attention_order=False):
    super().__init__()
    self.channels = channels
    if num_head_channels == -1:
          self.num_heads = num_heads
    else:
      assert (channels % num_head_channels == 0), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
      self.num_heads = channels // num_head_channels

    self.norm = normalization(channels)
    self.qkv = conv_nd(1 , channels , channels * 3 , 1)
    if use_new_attention_order:
            # split qkv before split heads
        self.attention = QKVAttention(self.num_heads)
    else:
            # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)
    self.proj_out = zero_module(conv_nd(1 , channels , channels , 1))
    
  def forward(self , x):
    b, c, *spatial = x.shape
    x = x.reshape(b, c, -1)
    qkv = self.qkv(self.norm(x))
    h = self.attention(qkv)
    h = self.proj_out(h)
    return (x + h).reshape(b, c, *spatial)
    
      


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    QKV= Query , Key , Value
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        #print("qkv shape values \n\t bs: {0}, width: {1}, length: {2}, ch: {3}, type:{4}, num_heads: {5}".format(bs , width , length, ch , type(qkv), self.n_heads))
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class TimeUnet(nn.Module):
  def __init__(self , image_size , in_channels , model_channels , out_channels , num_res_blocks, attention_resolutions , channel_mult = (1 , 2 , 4 , 8), conv_resample=True, dims=2 , num_head = 1, num_head_channels=-1, num_heads_upsample=-1, dropout=0.5,use_scale_shift_norm=False,resblock_updown=False,use_new_attention_order=False):
    super().__init__()
    self.image_size = image_size
    self.in_channels = in_channels
    self.model_channel = model_channels
    self.out_channels = out_channels
    self.attention_resolutions = attention_resolutions
    self.num_res_blocks = num_res_blocks
    self.channel_mult = channel_mult
    self.conv_resample = conv_resample
    self.num_head = num_head
    self.num_head_channels = num_head_channels
    self.num_heads_upsample = num_heads_upsample
    self.time_embed_dim = model_channels * 4
    self.time_embed = nn.Sequential(
      linear(self.model_channel , self.time_embed_dim ), nn.SiLU() ,
      linear(self.time_embed_dim , self.time_embed_dim))
    self.resblock_updown=resblock_updown
    
    ch = input_ch = int(channel_mult[0] * model_channels)
    self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
    )
    self._feature_size = ch
    input_block_chans= [ch]
    ds = 1
    for level , mult in enumerate(channel_mult):
      for _ in range(num_res_blocks):
        layers = [
           ResBlock(
                    ch,
                    self.time_embed_dim,
                    dropout,
                    out_channels=int(mult * model_channels),
                    dims=dims,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
              ]
        ch = int(mult * model_channels)
        if ds in attention_resolutions:
          layers.append(AttentionBlock( ch,
                                      num_heads=num_head,
                                      num_head_channels=num_head_channels,
                                      use_new_attention_order=use_new_attention_order)
                        )
        self.input_blocks.append(TimestepEmbedSequential(*layers))
        self._feature_size += ch
        input_block_chans.append(ch)
      if level != len(channel_mult) - 1:
        out_ch = ch
        self.input_blocks.append(
          TimestepEmbedSequential(ResBlock(ch , self.time_embed_dim , dropout , out_ch , dims, use_scale_shift_norm=use_scale_shift_norm ,down=True) 
          if resblock_updown 
          else Downsample(
            ch , use_conv=self.conv_resample , dims=dims , out_channels=out_ch
            )
          ) 
        )
        ch = out_ch
        input_block_chans.append(ch)
        ds *= 2
        self._feature_size += ch

    self.middle_blocks = TimestepEmbedSequential(
      ResBlock(ch , self.time_embed_dim, dropout, dims=dims , use_checkpoints=False , use_scale_shift_norm=use_scale_shift_norm),
      AttentionBlock(channels=ch , num_heads=num_head , num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order),
      ResBlock(ch , self.time_embed_dim , dropout , dims=dims , use_checkpoints=False , use_scale_shift_norm=use_scale_shift_norm),
    )
    self._feature_size += ch
    self.output_blocks = nn.ModuleList([])
    for level , mult in list(enumerate(channel_mult))[::-1]:
      for i in range(num_res_blocks+1):
        ich = input_block_chans.pop()
        layers = [
                    ResBlock(ch + ich,self.time_embed_dim,dropout,out_channels=int(model_channels * mult),dims=dims,use_scale_shift_norm=use_scale_shift_norm)
                ]
        ch = int(model_channels * mult)
        if ds in attention_resolutions:
                  layers.append(
                      AttentionBlock(ch,num_heads=num_heads_upsample,num_head_channels=num_head_channels,use_new_attention_order=use_new_attention_order,)
                  )
        if level and i == num_res_blocks:
          out_ch = ch
          layers.append(ResBlock(ch , self.time_embed_dim , dropout , out_channels=out_ch , dims=dims , use_scale_shift_norm=use_scale_shift_norm, up=True) 
                        if resblock_updown else Upsample(ch , use_conv=conv_resample,dims=dims, out_channels=out_ch ))
          ds //= 2
        self.output_blocks.append(TimestepEmbedSequential(*layers))
        self._feature_size += ch
        
    self.out = nn.Sequential(
      normalization(ch),
      nn.SiLU(),
      zero_module(conv_nd(dims , input_ch , out_channels , 3 , padding=1)),
    )
    
  def forward(self , x , timesteps):
    """
    Apply the model to an input batch.
    :param x: an [N x C x ...] Tensor of inputs.
    :param timesteps: a 1-D batch of timesteps.
    :return: an [N x C x ...] Tensor of outputs.
    """
    out_module = None
    try:
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channel))
        h = x
        for module in self.input_blocks:
            out_module = module
            h = module(h, emb)
            hs.append(h)
        h = self.middle_blocks(h, emb)
        module_num = 0
        for module in self.output_blocks:
            out_module = module
            skip_tensor = hs.pop()
            h = th.cat([h, skip_tensor], dim=1)
            h = module(h, emb)
            module_num += 1 
        h = h.type(x.dtype)
        return self.out(h)
    except Exception as e:
        print("#"*70)
        print(out_module)
        print("module occured in #{0}".format(module_num))
        raise e
    


    
    
class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)
    

if __name__ == '__main__':
  upsample = Upsample(channels=64 , use_conv=True , dims=2 , out_channels=16)
  downsample = Downsample(channels=16 , use_conv=True , dims=2 , out_channels=64)
  encoder = TimeUnet(image_size=(128 , 128) , in_channels=1 , model_channels=256 , out_channels=1 , num_res_blocks=2 , attention_resolutions=(32,16,8))
  linspace = np.linspace(0 , 1 , 20)
  summary(encoder)