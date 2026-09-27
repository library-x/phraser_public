"""

@InProceedings{swinunet,
author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
booktitle = {Proceedings of the European Conference on Computer Vision Workshops(ECCVW)},
year = {2022}
}

@misc{cao2021swinunet,
      title={Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
      author={Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
      year={2021},
      eprint={2105.05537},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}

SOURCE: https://github.com/HuCaoFighting/Swin-Unet/
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (int): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, out_dim_embed, norm_layer=nn.RMSNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(input_resolution * dim, out_dim_embed, bias=False)
        self.norm = norm_layer(input_resolution * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, H, W, C = x.shape
        assert H == 5 and W % self.input_resolution == 0, f"x size is wrong. {H} {W}"

        x = x.reshape(B, H, W // self.input_resolution, self.input_resolution * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x



class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, out_dim_embed, norm_layer=nn.RMSNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim_embed = out_dim_embed

        self.expand = nn.Linear(dim, self.input_resolution * out_dim_embed, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        assert H == 5

        x = self.norm(x)

        x = self.expand(x)
        x = x.reshape(B, H, W * self.input_resolution, self.out_dim_embed)


        return x

class PatchExpand2(nn.Module):
    def __init__(self, input_resolution, dim, out_dim_embed, norm_layer=nn.RMSNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim_embed = out_dim_embed

        self.expand = nn.Linear(dim, self.input_resolution * out_dim_embed, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B,  W, C = x.shape

        x = self.norm(x)

        x = self.expand(x)
        x = x.reshape(B,  W * self.input_resolution, self.out_dim_embed)


        return x