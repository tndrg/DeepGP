# -*- coding: utf-8 -*-
"""
@author: Taiyu Zhu
"""

import torch.nn as nn
import torch
import math

from layers.Transformer_EncDec import EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer,ProbAttention
from layers.Embed import CHRDataEmbedding,DataEmbedding,DataEmbeddingVocab
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from models import BaseModel
from timm.models.layers import DropPath
from functools import partial

class FCN(nn.Module):
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=256, 
                 hidden_layers=2, 
                 dropout=0.1,
                 activation='relu'): 
        super(FCN, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
        atten = False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.atten = atten
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        self.gradients = None

    def save_gradients(self, attn_gradients):
        self.gradients = attn_gradients

    def get_gradients(self):
        return self.gradients
    
    def forward(
        self, hidden_states, residual= None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        if self.atten:
            residual.register_hook(self.save_gradients)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
    atten = False
):

    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        atten = atten
    )
    block.layer_idx = layer_idx
    return block

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class SNPCOV_Model_mamba(BaseModel.BaseBinaryCov):

    def __init__(self, configs):
        super().__init__(configs)
        encoders = []
        dataEmbed = []
        norm_layers = []
        d_intermediate = 0
        self.l2 = configs.l2
        if configs.snp_embed=='cov':
            de_func = DataEmbedding
        elif configs.snp_embed=='vocab':
            de_func = DataEmbeddingVocab
        if configs.bi_direct:
            bimamba_type="v2"
        else:
            bimamba_type="v1"

        for i,pos in zip(configs.enc_len_chr,configs.pos):
            encoders.append(nn.ModuleList([create_block(configs.d_model,bimamba_type =bimamba_type )for i in range(configs.e_layers)]))
            dataEmbed.append(de_func(configs.enc_in, configs.d_model,pos,configs.dropout))
            norm_layers.append(RMSNorm(configs.d_model, eps=1e-5))


        self.apply(
            partial(
                _init_weights,
                n_layer=configs.e_layers,
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

        self.encoders = nn.ModuleList(encoders)
        self.dataEmbed = nn.ModuleList(dataEmbed)
        self.norm_f = nn.ModuleList(norm_layers)
        self.mhsa =  EncoderLayer(AttentionLayer(FullAttention(mask_flag=False),d_model=configs.d_model,n_heads=configs.n_heads),
                                  d_model=configs.d_model)

        if configs.use_cov:
            if not configs.use_pcs:
                configs.cov_dim = 2
            self.covariate_encoder = FCN(configs.cov_dim,configs.d_model)
            self.cov_norm = RMSNorm(configs.d_model, eps=1e-5)
            self.combine = FCN(23*configs.d_model, 1, hidden_layers = configs.n_fcn)    # 22 chorms + 1 cov
        else:
            self.combine = FCN(22*configs.d_model, 1, hidden_layers = configs.n_fcn)  
        
        
        self.dm = configs.dm
        self.final_pool = configs.final_pool
        self.use_pcs = configs.use_pcs
        self.use_cov = configs.use_cov
        self.fused_add_norm = configs.fused_add_norm
        

    def forward(self, x_all):
        x_snp, x_cov = x_all[:22],x_all[-1]
        if not self.use_pcs:
            x_cov = x_cov[:,:2]
        outputs = []
        for i in range(22):
            hidden_states = self.dataEmbed[i](x_snp[i])
            residual = None
            for layer in self.encoders[i]:
                hidden_states, residual = layer(hidden_states, residual)
            if not self.fused_add_norm:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm_f[i](residual.to(dtype=self.norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = layer_norm_fn(
                    hidden_states,
                    self.norm_f[i].weight,
                    self.norm_f[i].bias,
                    eps=self.norm_f[i].eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=True,
                    is_rms_norm=isinstance(self.norm_f[i], RMSNorm)
                )                
                # hidden_states = self.norm_f[i](hidden_states)+residual

            if self.final_pool == 'last':
                out = hidden_states[:,-1,:]
            elif self.final_pool == 'mean' or self.final_pool == 'atten':
                out = torch.mean(hidden_states,1)
                
            outputs.append(out)



        if self.use_cov:
            cov = self.covariate_encoder(x_cov)
            cov =self.cov_norm(cov)
            outputs.append(cov)
        if self.final_pool == 'atten':
            outputs, _ = self.mhsa(torch.stack(outputs,dim=1))
            x = outputs.view(outputs.size(0), -1)
        else:
            x = torch.cat(outputs, dim=1)
        x = self.opt(self.combine(x))
        return x
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2)


