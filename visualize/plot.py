import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch
from pathlib import Path

@torch.no_grad()
def pca_with_upsample(x, dim=3, factor=16):
    """
    Args: 
        x: tensor of shape [B,C,H,W]
        dim: target dimension for PCA
        factor: scale factor for upsampling
    Returns:
        tensor of shape [B,dim,H,W]
    """
    if factor != 1:
        x = F.interpolate(x, scale_factor=(factor, factor), mode="bilinear")
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1).reshape(-1, C)
    x = PCA(n_components=dim).fit_transform(x)
    x = torch.from_numpy(x).reshape(B, H, W, dim).permute(0, 3, 1, 2)
    return x

@torch.no_grad()
def plot_feat(feat, use_upsample=False, scale_factor=1, channels=[], save_path=None):
    """
    Args:
        feat: tensor of shape [C,H,W]
        scale_factor: scale factor for feature upsampling
        save_path: path to save figure
    """
    assert len(feat.shape) == 3, f"feat must be of shape [C,H,W], got {feat.shape}"
    
    if channels:
        feat = feat[channels, :, :]

    if use_upsample and scale_factor is not 1:
        feat = pca_with_upsample(feat.unsqueeze(0), dim=3, factor=scale_factor).squeeze(0)
    feat = (feat - feat.min()) / (feat.max() - feat.min())

    plt.axis("off")
    plt.title(f"{Path(save_path).stem if save_path else 'Feature Visualization'}")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

@torch.no_grad()
def plot_3feats(
    image, 
    teacher_feat, 
    student_feat,
    use_upsample=False,
    scale_factor=16,
    channels=[],
    save_dir=None
):
    """
    Args:
        image: raw image tensor of shape [C_1,H,W]
        teacher_feat: feature tensor from teacher model of shape [C_2,H,W]
        student_feat: feature tensor from student model of shape [C_2,H,W]
        scale_factor: scale factor for feature upsampling
        save_dir: directory to save figure
        channels: list of picked channels of raw image
    """
    if save_dir:
        plot_feat(image, channels, save_path=save_dir+'/raw_feat.png')
        plot_feat(teacher_feat, use_upsample, scale_factor, save_dir+'/teacher_feat.png')
        plot_feat(student_feat, use_upsample, scale_factor, save_dir+'/student_feat.png')
    else:
        plot_feat(image)
        plot_feat(teacher_feat, use_upsample, scale_factor=scale_factor)
        plot_feat(student_feat, use_upsample, scale_factor=scale_factor)