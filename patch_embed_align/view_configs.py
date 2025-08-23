import os
from omegaconf import OmegaConf

os.chdir('/data1/yangquan/DOFA-pytorch')
cfg = OmegaConf.load('src/configs/model/dca_seg.yaml')

print(f"Teacher: {cfg.pretrained_teacher_path}")
print(f"Student: {cfg.pretrained_student_path}")