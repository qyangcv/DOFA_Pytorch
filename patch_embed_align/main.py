import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

import hydra
from omegaconf import OmegaConf, DictConfig
from src.datasets.geobench_wrapper import GeoBenchDataset

# Add src directory to Python path for importing DOFA models
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.foundation_models.DOFA.models_dwv_seg import vit_base_patch16
from transformers import CLIPVisionModel
from src.datasets.data_module import BenchmarkDataModule


def patch_embed_align(ofa_model, clip_model, input_images, wave_list):
    """
    Align OFAViT and CLIP vision encoder patch_embed layer outputs and calculate loss
    
    Args:
        ofa_model: Pre-trained OFAViT model
        clip_model: CLIP vision encoder model  
        input_images: Input image tensor [B, C, H, W]
        wave_list: List of wavelengths required by OFA model
        
    Returns:
        loss: Alignment loss between the two patch_embed outputs
        ofa_patch_embed: OFAViT patch embedding output
        clip_patch_embed: CLIP patch embedding output
    """
    
    # Set models to evaluation mode
    ofa_model.eval()
    clip_model.eval()
    
    with torch.no_grad():
        # Get OFAViT patch embedding output
        # OFAViT uses Dynamic_MLP_OFA as patch_embed
        wavelist = torch.tensor(wave_list, device=input_images.device).float()
        ofa_patch_embed, _ = ofa_model.patch_embed(input_images, wavelist)
        # ofa_patch_embed shape: [B, num_patches, embed_dim]
        
        # Get CLIP vision encoder patch embedding output
        # CLIP typically requires RGB images, so take the first 3 channels
        rgb_images = input_images[:, :3, :, :]  # Take RGB channels
        
        # Use CLIP model's vision model to get patch embeddings
        clip_outputs = clip_model(pixel_values=rgb_images)
        # CLIP's patch embedding is usually in the last hidden states, excluding CLS token
        clip_patch_embed = clip_outputs.last_hidden_state[:, 1:, :]  # Remove CLS token
        # clip_patch_embed shape: [B, num_patches, clip_embed_dim]
        
        # Dimension alignment processing
        # If dimensions differ, need projection alignment
        ofa_dim = ofa_patch_embed.shape[-1]
        clip_dim = clip_patch_embed.shape[-1]
        
        if ofa_dim != clip_dim:
            # Create linear projection layer to project CLIP features to OFA dimension
            projection = nn.Linear(clip_dim, ofa_dim).to(input_images.device)
            clip_patch_embed_projected = projection(clip_patch_embed)
        else:
            clip_patch_embed_projected = clip_patch_embed
            
        # Ensure consistent number of patches
        ofa_patches = ofa_patch_embed.shape[1]
        clip_patches = clip_patch_embed_projected.shape[1]
        
        if ofa_patches != clip_patches:
            # If patch counts differ, adjust with interpolation
            min_patches = min(ofa_patches, clip_patches)
            ofa_patch_embed = ofa_patch_embed[:, :min_patches, :]
            clip_patch_embed_projected = clip_patch_embed_projected[:, :min_patches, :]
        
        # Calculate alignment loss
        # Use MSE loss to compute distance between two patch embeddings
        mse_loss = F.mse_loss(ofa_patch_embed, clip_patch_embed_projected)
        
        # Calculate cosine similarity loss (optional additional alignment objective)
        ofa_norm = F.normalize(ofa_patch_embed, p=2, dim=-1)
        clip_norm = F.normalize(clip_patch_embed_projected, p=2, dim=-1)
        cosine_sim = F.cosine_similarity(ofa_norm, clip_norm, dim=-1).mean()
        cosine_loss = 1 - cosine_sim  # Convert similarity to loss
        
        # Total loss: MSE loss + cosine loss
        total_loss = mse_loss + 0.1 * cosine_loss
        
    return total_loss, ofa_patch_embed, clip_patch_embed_projected


def load_models():
    """
    Load pre-trained OFAViT and CLIP models
    
    Returns:
        ofa_model: OFAViT model with loaded weights
        clip_model: CLIP vision encoder model
    """
    
    # Load OFAViT model
    ofa_model = vit_base_patch16()
    
    # Load pre-trained weights
    checkpoint_path = "../checkpoints/pretrained/DOFA_ViT_base_e100_full_weight.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        ofa_model.load_state_dict(checkpoint['model'], strict=False)
        print("OFAViT model weights loaded successfully")
    else:
        print(f"Warning: Pre-trained weight file not found: {checkpoint_path}")
    
    # Load CLIP vision encoder
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    print("CLIP vision encoder model loaded successfully")
    
    return ofa_model, clip_model


def main():
    """
    Main function: Demonstrate patch embedding alignment process
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    ofa_model, clip_model = load_models()
    ofa_model.to(device)
    clip_model.to(device)
    
    # Create example input
    batch_size = 2
    channels = 6  # OFA model supports multi-spectral input
    height, width = 224, 224
    
    # Generate random input images
    input_images = torch.randn(batch_size, channels, height, width).to(device)
    
    # Wavelength list required by OFA model (example)
    wave_list = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5]  # Center wavelengths for 6 bands
    
    print("Starting patch embedding alignment...")
    
    # Execute alignment and calculate loss
    try:
        loss, ofa_embed, clip_embed = patch_embed_align(
            ofa_model, clip_model, input_images, wave_list
        )
        
        print(f"Alignment loss: {loss.item():.6f}")
        print(f"OFA patch embedding shape: {ofa_embed.shape}")
        print(f"CLIP patch embedding shape: {clip_embed.shape}")
        
        # Output some statistics
        print(f"OFA embedding mean: {ofa_embed.mean().item():.6f}")
        print(f"CLIP embedding mean: {clip_embed.mean().item():.6f}")
        print(f"OFA embedding std: {ofa_embed.std().item():.6f}")
        print(f"CLIP embedding std: {clip_embed.std().item():.6f}")
        
        print("Patch embedding alignment completed!")
        
    except Exception as e:
        print(f"Error during alignment process: {str(e)}")
        import traceback
        traceback.print_exc()

@hydra.main(config_path="../src/configs/dataset", config_name="geobench_cashew")
def load_dataset(cfg: DictConfig):
    """
    Load m-cashew-plant dataset using Hydra configuration
    
    Returns:
        dataset: GeoBench dataset instance
        config: Dataset configuration
    """
    data_module = BenchmarkDataModule(
        dataset_config=cfg,
        batch_size=8,
        num_workers=4,
        pin_memory=True
    )
    data_module.setup()

    dataset_train = data_module.dataset_train
    dataset_val = data_module.dataset_val
    dataset_test = data_module.dataset_test

    # sample, mask = dataset_train[0]

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # # Get wavelength info from GeoBench task
    # geobench_wrapper = GeoBenchDataset(cfg)
    # task = geobench_wrapper.tasks[cfg.dataset_name] 
    # raw_dataset = task.get_dataset(split="train", transform=None, band_names=cfg.band_names)
    # raw_sample = raw_dataset[0]
    # wavelengths = [raw_sample.get_band_info(name).wavelength for name in (cfg.band_names or raw_sample.band_names)]
    # print(f"Wavelengths (μm): {wavelengths}")
    
    for batch_idx, (images, mask) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Image shape: {images.shape}, Mask shape: {mask.shape}")
        break

    return dataset_train, dataset_val, dataset_test

if __name__ == "__main__":
    load_dataset()
    # ofa_model, clip_model = load_models()  