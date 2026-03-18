import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg
import math

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp
from utils.evaluation import test_model_performance

from models.DWT.WTConv1d import WTConv1d


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, L):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, L)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, L):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, L))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, ca_num_heads=4, ca_conv_expand=1, ca_conv_expand_flag=True, sa_num_heads=8, qkv_bias=False, qk_scale=None, 
                       attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()
        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if ca_attention == 1:
            assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
            self.split_groups=self.dim//ca_num_heads
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(self.ca_num_heads):
                if not ca_conv_expand_flag:
                    local_conv = nn.Conv1d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.ca_num_heads) # original
                else:
                    local_conv = nn.Conv1d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2*ca_conv_expand), padding=(1+i*ca_conv_expand), stride=1, groups=dim//self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)
            self.proj0 = nn.Conv1d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups)
            self.bn = nn.BatchNorm1d(dim * expand_ratio)
            self.proj1 = nn.Conv1d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)
        else:
            assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."
            head_dim = dim // sa_num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, L):
        B, N, C = x.shape
        if self.ca_attention == 1:
            v = self.v(x)
            s = self.s(x).reshape(B, L, self.ca_num_heads, C//self.ca_num_heads).permute(2, 0, 3, 1)
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i= s[i]
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, L)
                if i == 0:
                    s_out = s_i
                else:
                    s_out = torch.cat([s_out,s_i], 2)
            s_out = s_out.reshape(B, C, L)
            s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
            self.modulator = s_out
            s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
            x = s_out * v
        else:
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
                self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B,C, L)).view(B, C, N).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, ca_num_heads, ca_conv_expand, ca_conv_expand_flag, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                    use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1, expand_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            ca_num_heads=ca_num_heads, ca_conv_expand=ca_conv_expand, ca_conv_expand_flag=ca_conv_expand_flag, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention, 
            expand_ratio=expand_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones(dim), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, L):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), L))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), L))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, sig_size=128, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, L = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, L
    

class NoneOverlapPatchEmbed(nn.Module):
    def __init__(self, sig_size=128, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, L = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, L


class HeadEmbedding(nn.Module):
    def __init__(self, head_conv, dim, norm_layer=nn.LayerNorm):
        super(HeadEmbedding, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=dim, kernel_size=head_conv, stride=1, padding=(head_conv - 1) // 2),
        )
        self.norm = norm_layer(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        _, _, L = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, L


class Sinusoidal(nn.Module):
    def __init__(self, d_model, max_len):
        super(Sinusoidal, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, requires_grad=False)
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x:torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        return self.encoding[:seq_len, :].unsqueeze(0).to(x.device)


class SMT(nn.Module):
    def __init__(self, sig_size=128, num_classes=11, head_conv=7, embed_dims=[64, 128, 512], ca_num_heads=[4, 4, -1], 
                 ca_conv_expand=[4, 2, -1], ca_conv_expand_flag = True, sa_num_heads=[-1, -1, 16], mlp_ratios=[8, 6, 2], 
                 qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, drop_rate=0., 
                 attn_drop_rate=0.2, drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                 depths=[1, 2, 1], mix_attentins = [0, 1, 0], ca_attentions=[1, 1, 0], expand_ratio=2, 
                 scaled_positional_emb = True, overlap_emb = True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(embed_dims)
        if scaled_positional_emb:
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            self.alpha = None
        self.pos_embed = Sinusoidal(d_model=embed_dims[0], max_len=sig_size)
        self.downsample_layers = nn.ModuleList()
        stem = HeadEmbedding(head_conv, embed_dims[0], norm_layer)
        self.downsample_layers.append(stem)
        for i in range(1, self.num_stages):
            if overlap_emb:
                downsample_layer = OverlapPatchEmbed(sig_size=sig_size if i == 0 else sig_size // (2 ** (i + 1)),
                                            patch_size=3,
                                            stride=2,
                                            in_chans=embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            else:
                downsample_layer = NoneOverlapPatchEmbed(sig_size=sig_size if i == 0 else sig_size // (2 ** (i + 1)),
                                            patch_size=3,
                                            stride=2,
                                            in_chans=embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        self.norm = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            stage = nn.ModuleList([Block(
                dim=embed_dims[i], ca_num_heads=ca_num_heads[i], ca_conv_expand=ca_conv_expand[i], ca_conv_expand_flag = ca_conv_expand_flag,
                    sa_num_heads=sa_num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                    use_layerscale=use_layerscale,
                    layerscale_value=layerscale_value,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                    ca_attention=0 if mix_attentins[i]==1 and j%2!=0 else ca_attentions[i], expand_ratio=expand_ratio)
                    for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
            norm = norm_layer(embed_dims[i])
            self.norm.append(norm)
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            x, L = self.downsample_layers[i](x)
            if i == 0 and self.alpha is not None:
                x += self.alpha * self.pos_embed(x)
            stage = self.stages[i]
            for blk in stage:
                x = blk(x, L)
            x = self.norm[i](x)
            if i != self.num_stages - 1:
                x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def smt_128(num_classes:int, sig_size:int):
    dim = 16
    model = SMT(
        head_conv=7, embed_dims=[dim, 2 * dim, 4 * dim], ca_num_heads=[4, 4, -1], ca_conv_expand=[4, 2, -1], sa_num_heads=[-1, 8, 16], mlp_ratios=[2, 2, 2],
        qkv_bias=True, depths=[1, 2, 1], ca_attentions=[1, 1, 0], mix_attentins=[0, 1, 0], expand_ratio=2, use_layerscale=True, num_classes=num_classes,
        sig_size=sig_size)
    return model


def smt_1024(num_classes:int, sig_size:int):
    dim = 16
    model = SMT(
        head_conv=7, embed_dims=[dim, 2 * dim, 4 * dim, 8 * dim], ca_num_heads=[4, 2, 2, -1], ca_conv_expand=[4, 2, 1, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[2, 2, 2, 2],
        qkv_bias=True, depths=[1, 1, 2, 1], ca_attentions=[1, 1, 1, 0], mix_attentins=[0, 0, 1, 0], expand_ratio=2, use_layerscale=True, num_classes=num_classes,
        sig_size=sig_size)
    return model


# without scaled positional embedding
def smt_128_SPE(num_classes:int, sig_size:int):
    dim = 16
    model = SMT(
        head_conv=7, embed_dims=[dim, 2 * dim, 4 * dim], ca_num_heads=[4, 4, -1], ca_conv_expand=[4, 2, -1], sa_num_heads=[-1, 8, 16], mlp_ratios=[2, 2, 2],
        qkv_bias=True, depths=[1, 2, 1], ca_attentions=[1, 1, 0], mix_attentins=[0, 1, 0], expand_ratio=2, use_layerscale=True, num_classes=num_classes,
        sig_size=sig_size, scaled_positional_emb=False)
    return model


# without overlap embedding
def smt_128_OE(num_classes:int, sig_size:int):
    dim = 16
    model = SMT(
        head_conv=7, embed_dims=[dim, 2 * dim, 4 * dim], ca_num_heads=[4, 4, -1], ca_conv_expand=[4, 2, -1], sa_num_heads=[-1, 8, 16], mlp_ratios=[2, 2, 2],
        qkv_bias=True, depths=[1, 2, 1], ca_attentions=[1, 1, 0], mix_attentins=[0, 1, 0], expand_ratio=2, use_layerscale=True, num_classes=num_classes,
        sig_size=sig_size, overlap_emb=False)
    return model


# without self attention
def smt_128_SAM(num_classes:int, sig_size:int):
    dim = 16
    model = SMT(
        num_stages=3, head_conv=7, embed_dims=[dim, 2 * dim, 4 * dim], ca_num_heads=[4, 4, 2], ca_conv_expand=[4, 2, 1], sa_num_heads=[-1, 8, 16], mlp_ratios=[2, 2, 2], 
        qkv_bias=True, depths=[1, 2, 1], ca_attentions=[1, 1, 1], mix_attentins=[0, 0, 0], expand_ratio=2, use_layerscale=True, num_classes=num_classes,
        sig_size=sig_size)
    return model


# without scale self attention
def smt_128_MSA(num_classes:int, sig_size:int):
    dim = 16
    model = SMT(
        num_stages=3, head_conv=7, embed_dims=[dim, 2 * dim, 4 * dim], ca_num_heads=[4, 4, -1], ca_conv_expand=[4, 2, -1], sa_num_heads=[4, 8, 16], mlp_ratios=[2, 2, 2], 
        qkv_bias=True, depths=[1, 2, 1], ca_attentions=[0, 0, 0], mix_attentins=[0, 0, 0], expand_ratio=2, use_layerscale=True, num_classes=num_classes,
        sig_size=sig_size)
    return model

def smt_128_naive(num_classes:int, sig_size:int):
    dim = 16
    model = SMT(
        head_conv=7, embed_dims=[dim, 2 * dim, 4 * dim], ca_num_heads=[4, 4, -1], ca_conv_expand=[4, 2, -1], ca_conv_expand_flag = False, sa_num_heads=[-1, 8, 16], mlp_ratios=[2, 2, 2],
        qkv_bias=True, depths=[1, 2, 1], ca_attentions=[1, 1, 0], mix_attentins=[0, 1, 0], expand_ratio=2, use_layerscale=True, num_classes=num_classes,
        sig_size=sig_size)
    return model


if __name__ == '__main__':
    input_data = torch.randn((1, 2, 128))
    model = smt_128(11, 128)
    test_model_performance(model, input_data)
    # input_data = torch.randn((1, 2, 1024))
    # model = smt_1024(24, 1024)
    # test_model_performance(model, input_data)
    input_data = torch.randn((1, 2, 128))
    model = smt_128_SPE(11, 128)
    test_model_performance(model, input_data)
    input_data = torch.randn((1, 2, 128))
    model = smt_128_OE(11, 128)
    test_model_performance(model, input_data)
    input_data = torch.randn((1, 2, 128))
    model = smt_128_SAM(11, 128)
    test_model_performance(model, input_data)
    input_data = torch.randn((1, 2, 128))
    model = smt_128_MSA(11, 128)
    test_model_performance(model, input_data)
    input_data = torch.randn((1, 2, 128))
    model = smt_128_naive(11, 128)
    test_model_performance(model, input_data)