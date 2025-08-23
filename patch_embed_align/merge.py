# merge student and teacher pth file
import torch
import os
from pathlib import Path

def merge_pretrained_models():
    """Merge teacher and student pretrained models into a single checkpoint"""
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    student_path = base_dir / "checkpoints" / "base" / "dofa_patch_embedding_p16_e100.pth"
    teacher_path = base_dir / "checkpoints" / "base" / "clipvision_patch_embedding_p16.pth"
    output_path = base_dir / "checkpoints" / "base" / "dca.pth"
    
    print(f"Loading teacher model from: {teacher_path}")
    print(f"Loading student model from: {student_path}")
    
    # Load pretrained weights
    teacher_weights = torch.load(teacher_path, map_location='cpu')
    student_weights = torch.load(student_path, map_location='cpu')
    
    # Create merged state dict
    merged_state_dict = {}
    
    # Add teacher weights with 'teacher.' prefix
    for key, value in teacher_weights.items():
        merged_state_dict[f"teacher.{key}"] = value
    
    # Add student weights with 'student.' prefix  
    for key, value in student_weights.items():
        merged_state_dict[f"student.{key}"] = value
    
    # Save merged checkpoint
    print(f"Saving merged checkpoint to: {output_path}")
    torch.save(merged_state_dict, output_path)
    
    print(f"Successfully merged {len(teacher_weights)} teacher params and {len(student_weights)} student params")
    print(f"Total merged params: {len(merged_state_dict)}")

if __name__ == "__main__":
    merge_pretrained_models()