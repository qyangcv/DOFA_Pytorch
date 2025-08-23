import os
from omegaconf import OmegaConf

# Set working directory to project root
os.chdir('/data1/yangquan/DOFA-pytorch')

# Load and resolve config
cfg = OmegaConf.load('src/configs/model/dca_seg.yaml')
resolved_cfg = OmegaConf.to_yaml(cfg, resolve=True)

print("Resolved paths:")
print(f"Teacher: {cfg.pretrained_teacher_path}")
print(f"Student: {cfg.pretrained_student_path}")
print(f"Exist teacher: {os.path.exists(cfg.pretrained_teacher_path)}")
print(f"Exist student: {os.path.exists(cfg.pretrained_student_path)}")