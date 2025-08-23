import torch
from src.foundation_models.DCA.dca_seg import Teacher, Student, dca_base_patch16

# comfirm consistence of model keys and ckpt keys #
model = dca_base_patch16()
model_keys = model.state_dict().keys()

ckpt = torch.load('./checkpoints/base/dca.pth', map_location='cpu')
ckpt_keys = ckpt.keys()
print("Missing in model:", ckpt_keys - model_keys)
print("Missing in pth:", model_keys - ckpt_keys)


# check dofa's patch embedsding #
ckpt_path = '../checkpoints/base/dofa_patch_embedding_p16_e100.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
model = Teacher()
missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)


# check clip's patch embedding #
ckpt_path = '../checkpoints/base/clipvision_patch_embedding_p16.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
model = Student()
missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
