#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/24 17:12
# @Author  : wangjie
# dropkey2: take index to guide the dropkey process

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from openpcmae.utils import misc
from openpcmae.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from openpcmae.utils.logger import *
import random
from knn_cuda import KNN
from openpcmae.cpp.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from openpcmae.models.knn_modules import knn_point

# add pos_eb and pos
# take attentive tokens
# the mean value of probability is 1/N
def get_dropkey_matrix_fromfeat2_v1(tokens: torch.Tensor):
    """
    Args:
        tokens (torch.Tensor):  [B, N, C]
    Returns:
        occ_matrix (torch.Tensor): occurrence matrix
    """
    B, N, C = tokens.shape

    # lens_keep = N - drop_number    #   N' = N - drop_numbers
    _, indices_feat = torch.sort(tokens, dim=1, descending=True)      #   [B, N, C] select the highest token in each dimension
    indices_top1 = indices_feat[:, :2, :]        #   [B, C]
    indices_top1 = indices_top1.reshape(B, -1)       #   [B, 2*C]

    # Calculate the number of occurrences of each index
    occurrences = torch.zeros(B, N, dtype=torch.long).cuda()    #   [B, N]
    occurrences.scatter_add_(1, indices_top1, torch.ones_like(indices_top1))

    # occurrences = occurrences.softmax(dim=1)    #   [B, N]
    occurrences = occurrences / (C * 2)           #   translate to probability
    # occurrences = occurrences.softmax(dim=1)  # [B, N]
    return occurrences

# add pos_eb and pos
# take attentive tokens
# the mean value of probability is from 1/N to 'mean_prob_value'
def get_dropkey_matrix_fromfeat2_v1_mapping1(tokens: torch.Tensor, mean_prob_value: float=0.3, lower_bound: float=0.05, upper_bound: float=0.95):
    """
    Args:
        tokens (torch.Tensor):  [B, N, C]
    Returns:
        occ_matrix (torch.Tensor): occurrence matrix
    """
    B, N, C = tokens.shape

    # lens_keep = N - drop_number    #   N' = N - drop_numbers
    _, indices_feat = torch.sort(tokens, dim=1, descending=True)      #   [B, N, C] select the highest token in each dimension
    indices_top1 = indices_feat[:, :2, :]        #   [B, C]
    indices_top1 = indices_top1.reshape(B, -1)       #   [B, 2*C]

    # Calculate the number of occurrences of each index
    occurrences = torch.zeros(B, N, dtype=torch.long).cuda()    #   [B, N]
    occurrences.scatter_add_(1, indices_top1, torch.ones_like(indices_top1))

    # occurrences = occurrences.softmax(dim=1)    #   [B, N]
    occurrences = occurrences / (C * 2)           #   translate to probability
    occurrences = occurrences*N*mean_prob_value

    # apply clip to limit the values between 0 and 0.95
    occurrences = occurrences.clamp(lower_bound, upper_bound)
    return occurrences

# add pos_eb and pos
# take attentive tokens
# the mean value of probability is 1/N
def get_dropkey_matrix_fromfeat2_v1_mapping2(tokens: torch.Tensor):
    """
    Args:
        tokens (torch.Tensor):  [B, N, C]
    Returns:
        occ_matrix (torch.Tensor): occurrence matrix
    """
    B, N, C = tokens.shape

    # lens_keep = N - drop_number    #   N' = N - drop_numbers
    _, indices_feat = torch.sort(tokens, dim=1, descending=True)      #   [B, N, C] select the highest token in each dimension
    indices_top1 = indices_feat[:, :2, :]        #   [B, C]
    indices_top1 = indices_top1.reshape(B, -1)       #   [B, 2*C]

    # Calculate the number of occurrences of each index
    occurrences = torch.zeros(B, N, dtype=torch.long).cuda()    #   [B, N]
    occurrences.scatter_add_(1, indices_top1, torch.ones_like(indices_top1))

    # occurrences = occurrences.softmax(dim=1)    #   [B, N]
    occurrences = occurrences / (C * 2)           #   translate to probability

    import math
    def gaussian(x, mu, sigma):
        return 1 / (sigma * torch.sqrt(torch.tensor(2 * math.pi))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # 选择均值和标准差
    mu = 0
    sigma = 0.5
    occurrences = gaussian(occurrences, mu, sigma)
    occurrences = 1-occurrences

    # occurrences = occurrences.softmax(dim=1)  # [B, N]
    return occurrences


class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3      [B, np, ns, 3]
            -----------------
            feature_global : B G C      #   [B, np, C]
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)
        # _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):       #   Input: [B, N, C]    Output:[B, N, C]
        x = self.fc1(x)         #   [B, N, C']
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)         #   [B, N, C]
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):   #   Input:[B, N, C]     #   Output:[B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)      #   [3, B, num_head, N, C/num_head]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)        #   [B, num_head, N, C/num_head]]

        attn = (q @ k.transpose(-2, -1)) * self.scale       #   [B, num_head, N, N]
        attn = attn.softmax(dim=-1)                         #   [B, num_head, N, N]
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)     #   [B, N, C]
        x = self.proj(x)                                    #   [B, N, C]
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x



class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):            #   Input: x:[B, np, C]  pos:[B, np, C] return_token_num: np*mask_ratio
        for _, block in enumerate(self.blocks):             #   x:[B, np, C]
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel         [B, np*mask_ratio, C]
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
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
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape      #   [B, np, 3]
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])     #   [B, np]
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False):
        # generate mask         #   neighborhood: [B, np, ns, 3]    center: [B, np, 3]      Output: x_vis:[B, np*(1-mask_ratio), C]     bool_masked_pos:[B, np]
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C       [B, np, C=384]

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)     #   [B, np*(1-mask_ratio), C]
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)         #  actually, vis_center:  [B, np*(1-mask_ratio), 3]
        pos = self.pos_embed(masked_center)                                         #   [B, np*(1-mask_ratio), C]

        # transformer
        x_vis = self.blocks(x_vis, pos)                                             #   [B, np*(1-mask_ratio), C]
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos




class TransformerEncoder_hierarchical(nn.Module):
    def __init__(self, num_heads=12, drop_path_rate=0., num_group=128,
                 encoder_depths=[4, 4, 4],
                 encoder_dims=[384, 384, 384],
                 attn_drop_rates=[0.0, 0.0, 0.0]):
        super().__init__()
        self.encoder_depths = encoder_depths
        self.encoder_dims = encoder_dims
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rates = attn_drop_rates
        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(TransformerEncoder(
                embed_dim=self.encoder_dims[i],
                depth=self.encoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                num_heads=num_heads,
                attn_drop_rate=self.attn_drop_rates[i],
            ))
            depth_count += self.encoder_depths[i]

    def forward(self, x, pos_eb, pos):
        '''
            Input:  x: [B, N, C]
                    pos_eb: [B, N, C]
                    pos_: [B, N, 3]
            Output: x_remain_list: list, [B, N, C]
        '''
        B, N, C = x.shape
        x_remain_list = []

        for i in range(len(self.encoder_blocks)):
            pos = pos
            pos_eb = pos_eb
            x = self.encoder_blocks[i](x=x, pos=pos_eb)
            x_remain_list.append(x)

        return x_remain_list



# ---------------------------------------------------------------------

class Attention2_v1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,use_DropKey=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_rate = attn_drop
        self.use_DropKey = use_DropKey

    def forward(self, x):   #   Input:[B, N, C]     #   Output:[B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)      #   [3, B, num_head, N, C/num_head]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)        #   [B, num_head, N, C/num_head]]

        attn = (q @ k.transpose(-2, -1)) * self.scale       #   [B, num_head, N, N]

        m_r = get_dropkey_matrix_fromfeat2_v1(tokens=x) #   [B, N]
        m_r = m_r.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, N, 1) #   [B, num_head, N, N]
        # use DropKey as regularizer 10
        if self.use_DropKey == True:
            # m_r = torch.ones_like(attn) * self.attn_drop_rate
            attn = attn + torch.bernoulli(m_r) * -1e12
            attn = attn.softmax(dim=-1)  # [B, num_head, N, N]
        else:
            attn = attn.softmax(dim=-1)  # [B, num_head, N, N]
            attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)     #   [B, N, C]
        x = self.proj(x)                                    #   [B, N, C]
        x = self.proj_drop(x)
        return x


class Block2_v1(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_DropKey=False):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention2_v1(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_DropKey=use_DropKey)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder2_v1(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., use_DropKey=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block2_v1(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                use_DropKey=use_DropKey,
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerEncoder2_v1_hierarchical(nn.Module):
    def __init__(self, num_heads=12, drop_path_rate=0., num_group=128,
                 encoder_depths=[4, 4, 4],
                 encoder_dims=[384, 384, 384],
                 attn_drop_rates=[0.0, 0.0, 0.0],
                 use_DropKeys=[False, False, False],):
        super().__init__()
        self.encoder_depths = encoder_depths
        self.encoder_dims = encoder_dims
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rates = attn_drop_rates
        self.use_DropKeys = use_DropKeys
        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(TransformerEncoder2_v1(
                embed_dim=self.encoder_dims[i],
                depth=self.encoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                num_heads=num_heads,
                attn_drop_rate=self.attn_drop_rates[i],
                use_DropKey=self.use_DropKeys[i],
            ))
            depth_count += self.encoder_depths[i]

    def forward(self, x, pos_eb, pos):
        '''
            Input:  x: [B, N, C]
                    pos_eb: [B, N, C]
                    pos_: [B, N, 3]
            Output: x_remain_list: list, [B, N, C]
        '''
        B, N, C = x.shape
        x_remain_list = []

        for i in range(len(self.encoder_blocks)):
            pos = pos
            pos_eb = pos_eb
            x = self.encoder_blocks[i](x=x, pos=pos_eb)
            x_remain_list.append(x)

        return x_remain_list




# ---------------------------------------------------------------------

class Attention2_v1_mapping1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,use_DropKey=False,\
                 mean_prob_value=0.3, lower_bound=0.05, upper_bound=0.95):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_rate = attn_drop
        self.use_DropKey = use_DropKey
        self.mean_prob_value = mean_prob_value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, x):   #   Input:[B, N, C]     #   Output:[B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)      #   [3, B, num_head, N, C/num_head]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)        #   [B, num_head, N, C/num_head]]

        attn = (q @ k.transpose(-2, -1)) * self.scale       #   [B, num_head, N, N]

        m_r = get_dropkey_matrix_fromfeat2_v1_mapping1(tokens=x, mean_prob_value=self.mean_prob_value, lower_bound=self.lower_bound, upper_bound=self.upper_bound) #   [B, N]
        m_r = m_r.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, N, 1) #   [B, num_head, N, N]
        # use DropKey as regularizer 10
        if self.use_DropKey == True:
            # m_r = torch.ones_like(attn) * self.attn_drop_rate
            attn = attn + torch.bernoulli(m_r) * -1e12
            attn = attn.softmax(dim=-1)  # [B, num_head, N, N]
        else:
            attn = attn.softmax(dim=-1)  # [B, num_head, N, N]
            attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)     #   [B, N, C]
        x = self.proj(x)                                    #   [B, N, C]
        x = self.proj_drop(x)
        return x


class Block2_v1_mapping1(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_DropKey=False,\
                 mean_prob_value=0.3, lower_bound=0.05, upper_bound=0.95):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention2_v1_mapping1(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_DropKey=use_DropKey,
            mean_prob_value=mean_prob_value, lower_bound=lower_bound, upper_bound=upper_bound)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder2_v1_mapping1(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., use_DropKey=False,\
                 mean_prob_value=0.3, lower_bound=0.05, upper_bound=0.95):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block2_v1_mapping1(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                use_DropKey=use_DropKey,
                mean_prob_value=mean_prob_value, lower_bound=lower_bound, upper_bound=upper_bound,
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerEncoder2_v1_hierarchical_mapping1(nn.Module):
    def __init__(self, num_heads=12, drop_path_rate=0., num_group=128,
                 encoder_depths=[4, 4, 4],
                 encoder_dims=[384, 384, 384],
                 attn_drop_rates=[0.0, 0.0, 0.0],
                 use_DropKeys=[False, False, False],
                 mean_prob_values=[0.3, 0.3, 0.3],
                 lower_bound=0.05, upper_bound=0.95
                 ):
        super().__init__()
        self.encoder_depths = encoder_depths
        self.encoder_dims = encoder_dims
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rates = attn_drop_rates
        self.use_DropKeys = use_DropKeys
        self.mean_prob_values = mean_prob_values
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(TransformerEncoder2_v1_mapping1(
                embed_dim=self.encoder_dims[i],
                depth=self.encoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                num_heads=num_heads,
                attn_drop_rate=self.attn_drop_rates[i],
                use_DropKey=self.use_DropKeys[i],
                mean_prob_value=self.mean_prob_values[i],
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
            ))
            depth_count += self.encoder_depths[i]

    def forward(self, x, pos_eb, pos):
        '''
            Input:  x: [B, N, C]
                    pos_eb: [B, N, C]
                    pos_: [B, N, 3]
            Output: x_remain_list: list, [B, N, C]
        '''
        B, N, C = x.shape
        x_remain_list = []

        for i in range(len(self.encoder_blocks)):
            pos = pos
            pos_eb = pos_eb
            x = self.encoder_blocks[i](x=x, pos=pos_eb)
            x_remain_list.append(x)

        return x_remain_list

@MODELS.register_module()
class APCT_mapping1(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.attn_drop_rates = config.attn_drop_rates
        self.use_DropKeys = config.use_DropKeys
        self.mean_prob_values = config.mean_prob_values
        self.lower_bound = config.lower_bound
        self.upper_bound = config.upper_bound

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = TransformerEncoder2_v1_hierarchical_mapping1(
            encoder_depths=config.encoder_depths,
            encoder_dims=config.encoder_dims,
            num_heads=self.num_heads,
            num_group=self.num_group,
            drop_path_rate=self.drop_path_rate,
            attn_drop_rates=self.attn_drop_rates,
            use_DropKeys=self.use_DropKeys,
            mean_prob_values=self.mean_prob_values,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim, 256),
                # nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
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
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, return_pred_mask=False):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N


        pos = self.pos_embed(center)
        x = group_input_tokens

        # transformer
        x_list = self.blocks(x=x, pos_eb=pos, pos=center)  #            Output: x:[B, N, C] x_remained: [B, N-num_drop, C] x_removed: [B, num_drop, C]

        pred_list = []
        for i in range(len(x_list)):
            x_list_i = self.norm(x_list[i])
            x_max_i = x_list_i.max(1)[0]
            # pred_i = self.cls_head_finetune_blocks[i](x_max_i)    # [B, cls_dim]
            pred_i = self.cls_head_finetune(x_max_i)    # [B, cls_dim]
            pred_list.append(pred_i)

        if return_pred_mask:
            if len(pred_list) == 3:
                return pred_list[0], pred_list[1], pred_list[2]
            else:
                return pred_list
        else:   # fix: return the last one, not the first one
            return pred_list[-1]
