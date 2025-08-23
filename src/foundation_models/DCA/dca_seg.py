# Implementation of aligning patch-embedding feature of DOFA (multi-channels) and CLIP (RGB-channels)
import torch
import torch.nn as nn

from .wave_dynamic_layer import Dynamic_MLP_OFA


class Student(nn.Module):
    """Student model for dynamic-channel wavelengths encoding
    Input: 
    Return:
    """
    def __init__(
        self,
        image_size=224,
        embed_dim=768,
        patch_size=16,
        wv_planes=128, 
    ):
        super().__init__()
        self.wv_planes = wv_planes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.image_size = image_size

        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=self.wv_planes, inter_dim=128, kernel_size=self.patch_size, embed_dim=self.embed_dim
        )

    def forward(self, x, wv_lst):
        patch_embeds, _ = self.patch_embed(x, wv_lst)
        return patch_embeds


class Teacher(nn.Module):
    """Teacher model for rgb-channel wavelengths encoding
    Input:
    Return:
    """
    def __init__(
        self,
        image_size=224,
        embed_dim=768,
        patch_size=16,
        rgb_indices=[0, 1, 2]
    ):
        super().__init__()
        self.img_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False
        )

        self.rgb_indices = rgb_indices

    def get_rgb_feat(self, x, indices):
        return x[:, indices, :, :]

    def forward(self, x):
        x = self.get_rgb_feat(x, self.rgb_indices)
        patch_embeds = self.patch_embedding(x)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        return patch_embeds


class DCA(nn.Module):
    """Dynamic Channel Aligner
    Input:
    Return:
    """
    def __init__(
        self,
        image_size=224,
        embed_dim=768,
        patch_size=16,
        wv_planes=128,
        rgb_indices=[0,1,2]
    ):
        super().__init__()

        # clip
        self.teacher = Teacher(
            image_size=image_size,
            embed_dim=embed_dim,
            patch_size=patch_size,
            rgb_indices=rgb_indices,
            
        )

        # dofa
        self.student = Student(
            image_size=image_size,
            embed_dim=embed_dim,
            patch_size=patch_size,
            wv_planes=wv_planes,
        )

    def forward(self, x, wv_lst):
        student_out = self.student(x, wv_lst) 
        teacher_out = self.teacher(x)         
        return teacher_out, student_out
    

def dca_base_patch16():
    model = DCA(
        image_size=224,
        embed_dim=768,
        patch_size=16,
        rgb_indices=[2,1,0], # geobench_cashew的rgb通道索引
    )
    return model