import torch
from transformers import CLIPVisionModel
from torchgeo.models import dofa_base_patch16_224

# extract dofa's patch embedding weight
checkpoint_path = '../checkpoints/pretrained/DOFA_ViT_base_e100.pth'
ckpt = torch.load(checkpoint_path, map_location='cpu')

print("Available keys in checkpoint:")
for key in ckpt.keys():
    print(key)

print("\nPatch embedding keys:")
patch_embed_weights = {}
for key in ckpt.keys():
    if key.startswith('patch_embed'):
        print(key)
        patch_embed_weights[key.replace('patch_embed', 'patch_embedding')] = ckpt[key]

save_path = '../checkpoints/base/dofa_patch_embedding_p16_e100.pth'
torch.save(patch_embed_weights, save_path)
print(f"\nPatch embedding weights saved to: {save_path}")
print(f"Saved {len(patch_embed_weights)} patch embedding parameters")


# extract clip_vison_encoder's patch embedding weight
clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

# get patch embedding weights
patch_embed_weights = {}
for key, param in clip_model.state_dict().items():
    if key.startswith('vision_model.embeddings.patch_embedding'):
        patch_embed_weights['patch_embedding.weight'] = param

# save to file
save_path = '../checkpoints/base/clipvision_patch_embedding_p16.pth'
torch.save(patch_embed_weights, save_path)

print(f"Saved {len(patch_embed_weights)} patch embedding parameters to {save_path}")
for key in patch_embed_weights.keys():
    print(f"  {key}: {patch_embed_weights[key].shape}")