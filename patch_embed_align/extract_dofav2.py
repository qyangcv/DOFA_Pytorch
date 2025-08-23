import torch
import torch.nn as nn
import sys
import os

# Add the src directory to Python path to import DCA modules
# sys.path.append('../src')
from src.foundation_models.DCA.dca_seg import Teacher
from src.foundation_models.DCA.wave_dynamic_layer import Dynamic_MLP_OFA


def extract_patch_embedding_weights(checkpoint_path, output_path):
    """
    Extract patch embedding weights from DOFA checkpoint and save as Teacher model weights
    
    Args:
        checkpoint_path: Path to the original DOFA checkpoint
        output_path: Path to save the extracted Teacher model weights
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the original checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Filter patch embedding related weights
    patch_embed_weights = {}
    
    print("Extracting patch embedding weights...")
    for key, value in checkpoint.items():
        if key.startswith('patch_embed'):
            # Remove 'patch_embed.' prefix to match Teacher model structure
            new_key = key.replace('patch_embed.', '')
            patch_embed_weights[new_key] = value
            print(f"  Extracted: {key} -> {new_key} {value.shape}")
    
    print(f"\nExtracted {len(patch_embed_weights)} weight tensors")
    
    # Create a Teacher model instance to verify compatibility
    print("\nCreating Teacher model for validation...")
    teacher_model = Teacher(
        image_size=224,
        embed_dim=768,
        patch_size=16,
        wv_planes=128
    )
    
    # Create the state dict for the patch_embed module
    teacher_state_dict = {}
    for key, value in patch_embed_weights.items():
        teacher_state_dict[f'patch_embed.{key}'] = value
    
    # Verify that the weights are compatible
    print("\nValidating weight compatibility...")
    try:
        teacher_model.load_state_dict(teacher_state_dict, strict=False)
        print("✓ Weight shapes are compatible with Teacher model")
        
        # Check which weights were loaded and which are missing
        model_keys = set(teacher_model.state_dict().keys())
        loaded_keys = set(teacher_state_dict.keys())
        
        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys
        
        if missing_keys:
            print(f"Missing keys (will be randomly initialized): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (will be ignored): {unexpected_keys}")
            
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        return False
    
    # Save the extracted weights
    print(f"\nSaving extracted weights to: {output_path}")
    torch.save(teacher_state_dict, output_path)
    
    # Verify the saved file
    print("Verifying saved checkpoint...")
    saved_checkpoint = torch.load(output_path, map_location='cpu')
    print(f"Saved checkpoint contains {len(saved_checkpoint)} keys:")
    for key, value in saved_checkpoint.items():
        print(f"  {key}: {value.shape}")
    
    return True


def analyze_weight_mapping():
    """
    Analyze the weight mapping between DOFA checkpoint and Teacher model
    """
    print("Analyzing weight structure mapping...")
    print("="*60)
    
    # Load original checkpoint
    checkpoint_path = './checkpoints/pretrained/dofav2_vit_base_e150.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create Teacher model
    teacher_model = Teacher()
    teacher_state_dict = teacher_model.state_dict()
    
    print("DOFA checkpoint patch_embed keys:")
    for key in sorted(checkpoint.keys()):
        if key.startswith('patch_embed'):
            print(f"  {key}: {checkpoint[key].shape}")
    
    print("\nTeacher model keys:")
    for key in sorted(teacher_state_dict.keys()):
        print(f"  {key}: {teacher_state_dict[key].shape}")
    
    print("\nMapping compatibility:")
    for dofa_key in checkpoint.keys():
        if dofa_key.startswith('patch_embed'):
            teacher_key = dofa_key  # Direct mapping since structure should match
            if teacher_key in teacher_state_dict:
                dofa_shape = checkpoint[dofa_key].shape
                teacher_shape = teacher_state_dict[teacher_key].shape
                match = "✓" if dofa_shape == teacher_shape else "✗"
                print(f"  {match} {dofa_key} -> {teacher_key}")
                print(f"    DOFA: {dofa_shape}, Teacher: {teacher_shape}")


def test_teacher_model_loading(checkpoint_path):
    """
    Test loading the extracted weights into a Teacher model
    """
    print("Testing Teacher model loading...")
    print("="*60)
    
    # Create Teacher model
    teacher_model = Teacher(
        image_size=224,
        embed_dim=768,
        patch_size=16,
        wv_planes=128
    )
    
    # Load extracted weights
    extracted_weights = torch.load(checkpoint_path, map_location='cpu')
    
    # Load weights into model
    missing_keys, unexpected_keys = teacher_model.load_state_dict(extracted_weights, strict=False)
    
    print(f"Loaded {len(extracted_weights)} weight tensors")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        # Create dummy input
        batch_size = 1
        channels = 9
        height, width = 224, 224
        x = torch.randn(batch_size, channels, height, width)
        wv_lst = torch.tensor([0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865])  # Example wavelengths
        
        # Forward pass
        with torch.no_grad():
            output = teacher_model(x, wv_lst)
        
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Wavelengths shape: {wv_lst.shape}")
        print(f"  Output shape: {output[0].shape}")  # patch embeddings
        print(f"  Waves encoding shape: {output[1].shape}")  # wave encoding
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True


if __name__ == '__main__':
    # Paths
    checkpoint_path = './checkpoints/pretrained/dofav2_vit_base_e150.pth'
    output_path = './checkpoints/teacher_patch_embed_weights.pth'
    
    print("DOFA Patch Embedding Extraction Tool")
    print("="*60)
    
    # Step 1: Analyze weight mapping
    analyze_weight_mapping()
    
    print("\n" + "="*60)
    
    # Step 2: Extract and save weights
    success = extract_patch_embedding_weights(checkpoint_path, output_path)
    
    if success:
        print("\n" + "="*60)
        
        # Step 3: Test the extracted weights
        test_teacher_model_loading(output_path)
        
        print("\n" + "="*60)
        print("✓ Extraction completed successfully!")
        print(f"Teacher model weights saved to: {output_path}")
        print("\nUsage:")
        print("```python")
        print("from src.foundation_models.DCA.dca_seg import Teacher")
        print("teacher = Teacher()")
        print("teacher.load_state_dict(torch.load('checkpoints/teacher_patch_embed_weights.pth'))")
        print("```")
    else:
        print("✗ Extraction failed!")