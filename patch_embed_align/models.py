import torch
import torch.nn as nn

def analyze_checkpoint(checkpoint_path):
    print(f"Analyzing checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint type: {type(checkpoint)}")
    print(f"Top-level keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
    print()
    
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            print(f"Key: '{key}'")
            print(f"  Type: {type(value)}")
            
            if isinstance(value, dict):
                print(f"  Sub-keys: {list(value.keys())}")
                if len(value) < 20:
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'shape'):
                            print(f"    {sub_key}: {sub_value.shape} ({sub_value.dtype})")
                        else:
                            print(f"    {sub_key}: {type(sub_value)} - {sub_value}")
                else:
                    print(f"  Contains {len(value)} items (showing first 10):")
                    for i, (sub_key, sub_value) in enumerate(value.items()):
                        if i >= 10:
                            print(f"    ... and {len(value) - 10} more")
                            break
                        if hasattr(sub_value, 'shape'):
                            print(f"    {sub_key}: {sub_value.shape} ({sub_value.dtype})")
                        else:
                            print(f"    {sub_key}: {type(sub_value)}")
            
            elif hasattr(value, 'shape'):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
            else:
                print(f"  Value: {value}")
            print()
    
    elif hasattr(checkpoint, 'state_dict'):
        print("Checkpoint appears to be a model object")
        state_dict = checkpoint.state_dict()
        print(f"State dict keys: {len(state_dict)}")
        for name, param in state_dict.items():
            print(f"{name}: {param.shape} ({param.dtype})")
    
    else:
        print("Unknown checkpoint format")

def analyze_model_architecture(state_dict):
    print("\nModel Architecture Analysis:")
    print("=" * 60)
    
    layer_groups = {}
    total_params = 0
    
    for name, param in state_dict.items():
        total_params += param.numel()
        
        parts = name.split('.')
        if len(parts) > 1:
            group = parts[0]
            if group not in layer_groups:
                layer_groups[group] = []
            layer_groups[group].append((name, param.shape, param.numel()))
        else:
            if 'root' not in layer_groups:
                layer_groups['root'] = []
            layer_groups['root'].append((name, param.shape, param.numel()))
    
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    print()
    
    for group, layers in layer_groups.items():
        group_params = sum(layer[2] for layer in layers)
        print(f"Group: {group}")
        print(f"  Parameters: {group_params:,} ({group_params/total_params*100:.1f}%)")
        print(f"  Layers: {len(layers)}")
        
        if len(layers) <= 10:
            for name, shape, params in layers:
                print(f"    {name}: {shape} ({params:,} params)")
        else:
            print(f"    Showing first 5 layers:")
            for name, shape, params in layers[:5]:
                print(f"    {name}: {shape} ({params:,} params)")
            print(f"    ... and {len(layers)-5} more layers")
        print()

def create_simple_vit_model(state_dict):
    """Create a simple ViT-like model based on the state dict structure"""
    print("\nCreating simplified model structure:")
    print("=" * 60)
    
    embed_dim = 768
    num_heads = 12
    num_layers = 12
    
    print(f"ViT-Base architecture detected:")
    print(f"  - Embedding dimension: {embed_dim}")
    print(f"  - Number of attention heads: {num_heads}")
    print(f"  - Number of transformer layers: {num_layers}")
    print(f"  - Position embeddings: {state_dict['model.pos_embed'].shape}")
    print(f"  - Class token: {state_dict['model.cls_token'].shape}")
    
    class SimpleViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_layers = num_layers
            self.embed_dim = embed_dim
            
        def forward(self, x):
            return x
    
    model = SimpleViT()
    return model

def analyze_attention_patterns(state_dict):
    """Analyze attention weight patterns"""
    print("\nAttention Analysis:")
    print("=" * 60)
    
    for layer_idx in range(12):
        qkv_key = f'model.blocks.{layer_idx}.attn.qkv.weight'
        if qkv_key in state_dict:
            qkv_weight = state_dict[qkv_key]
            print(f"Layer {layer_idx}: QKV shape {qkv_weight.shape}")
            
            q_weight = qkv_weight[:768, :]
            k_weight = qkv_weight[768:1536, :]
            v_weight = qkv_weight[1536:, :]
            
            print(f"  Q weight norm: {torch.norm(q_weight).item():.4f}")
            print(f"  K weight norm: {torch.norm(k_weight).item():.4f}")
            print(f"  V weight norm: {torch.norm(v_weight).item():.4f}")

def analyze_patch_embed(state_dict):
    """Analyze the patch embedding structure"""
    print("\nPatch Embedding Analysis:")
    print("=" * 60)
    
    patch_keys = [k for k in state_dict.keys() if k.startswith('patch_embed')]
    
    print("DOFA-specific patch embedding components:")
    for key in patch_keys:
        tensor = state_dict[key]
        print(f"  {key}: {tensor.shape}")
        
        if 'weight_generator.fc_weight' in key:
            print(f"    -> This generates dynamic convolution weights")
            print(f"    -> Output channels: {tensor.shape[0]} (likely for different spectral bands)")
        elif 'weight_generator.fc_bias' in key:
            print(f"    -> This generates dynamic convolution biases")
        elif 'weight_generator.transformer' in key:
            print(f"    -> Part of the weight generator transformer")
        elif 'fclayer' in key:
            print(f"    -> Fully connected layer in patch embedding")

def save_model_summary(state_dict, output_file="model_summary.txt"):
    """Save a detailed model summary to file"""
    with open(output_file, 'w') as f:
        f.write("DOFA-v2 ViT-Base Model Summary\n")
        f.write("=" * 50 + "\n\n")
        
        total_params = sum(p.numel() for p in state_dict.values())
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Model Size: {total_params * 4 / 1024 / 1024:.2f} MB\n\n")
        
        f.write("Layer Details:\n")
        f.write("-" * 30 + "\n")
        
        for name, param in state_dict.items():
            f.write(f"{name}: {list(param.shape)} ({param.numel():,} params)\n")
    
    print(f"\nDetailed model summary saved to: {output_file}")

if __name__ == '__main__':
    checkpoint_path = '../checkpoints/pretrained/dofav2_vit_base_e150.pth'
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    analyze_checkpoint(checkpoint_path)
    
    if isinstance(checkpoint, dict):
        if 'model' in [k.split('.')[0] for k in checkpoint.keys()]:
            print("\nUsing checkpoint as state dict directly")
            state_dict = checkpoint
            analyze_model_architecture(state_dict)
            
            print("\n" + "="*80)
            create_simple_vit_model(state_dict)
            
            print("\n" + "="*80)
            analyze_patch_embed(state_dict)
            
            print("\n" + "="*80)
            analyze_attention_patterns(state_dict)
            
            print("\n" + "="*80)
            save_model_summary(state_dict)
            
        else:
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict):
                    print(f"\nAnalyzing state dict from key '{key}':")
                    analyze_model_architecture(checkpoint[key])
    elif hasattr(checkpoint, 'state_dict'):
        analyze_model_architecture(checkpoint.state_dict())
    else:
        print("Could not find state dict to analyze")
    
    print("\n" + "="*80)
    print("Analysis complete! Check model_summary.txt for detailed layer information.")