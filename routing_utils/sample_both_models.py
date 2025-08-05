#!/usr/bin/env python3
"""
Sample from both baseline and routing models for comparison
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import argparse

# Import both models
import model
import model_novel

def load_baseline_model(checkpoint_path, device):
    """Load baseline model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model from checkpoint
    model_args = checkpoint['model_args']
    
    # Remove routing-specific parameters from model_args for baseline model
    baseline_model_args = model_args.copy()
    baseline_model_args.pop('use_routing', None)
    baseline_model_args.pop('top_k_heads', None)
    baseline_model_args.pop('entropy_reg_coef', None)
    
    baseline_model = model.GPT(model.GPTConfig(**baseline_model_args))
    
    # Load state dict, filtering out routing-specific parameters if present
    state_dict = checkpoint['model']
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if not any(routing_key in key for routing_key in ['gate_net', 'routing']):
            filtered_state_dict[key] = value
    
    baseline_model.load_state_dict(filtered_state_dict, strict=False)
    return baseline_model

def load_routing_model(checkpoint_path, device):
    """Load routing model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model from checkpoint
    model_args = checkpoint['model_args']
    routing_model = model_novel.GPT(model_novel.GPTConfig(**model_args))
    
    # Load state dict
    routing_model.load_state_dict(checkpoint['model'])
    return routing_model

def get_encoder_decoder(checkpoint_path):
    """Get encoder/decoder functions from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    
    return encode, decode

def generate_text(model, x, max_new_tokens, temperature, top_k, device, ctx):
    """Generate text using the model."""
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            return y[0].tolist()

def main():
    parser = argparse.ArgumentParser(description='Generate text samples from both models')
    parser.add_argument('--baseline_checkpoint', type=str, 
                       default='out-enwik8-routing-ablation/ckpt.pt',
                       help='Path to baseline model checkpoint')
    parser.add_argument('--routing_checkpoint', type=str,
                       default='out-enwik8-routing/ckpt.pt',
                       help='Path to routing model checkpoint')
    parser.add_argument('--start', type=str, default='\n',
                       help='Starting text for generation')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to generate per model')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                       help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=200,
                       help='Top-k sampling')
    parser.add_argument('--seed', type=int, default=1337,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       help='Data type')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile')
    
    args = parser.parse_args()
    
    # Set up device and dtype
    device = args.device
    dtype = args.dtype
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load models
    print("Loading models...")
    try:
        baseline_model = load_baseline_model(args.baseline_checkpoint, device)
        baseline_model.eval()
        baseline_model.to(device)
        if args.compile:
            baseline_model = torch.compile(baseline_model)
        print(f"Baseline model loaded from {args.baseline_checkpoint}")
        print(f"Baseline model parameters: {sum(p.numel() for p in baseline_model.parameters()):,} (total with positional embeddings)")
        print(f"Baseline model parameters: {baseline_model.get_num_params():,} (non-embedding)")
    except Exception as e:
        print(f"Error loading baseline model: {e}")
        return
    
    try:
        routing_model = load_routing_model(args.routing_checkpoint, device)
        routing_model.eval()
        routing_model.to(device)
        if args.compile:
            routing_model = torch.compile(routing_model)
        print(f"Routing model loaded from {args.routing_checkpoint}")
        print(f"Routing model parameters: {sum(p.numel() for p in routing_model.parameters()):,} (total with positional embeddings)")
        print(f"Routing model parameters: {routing_model.get_num_params():,} (non-embedding)")
    except Exception as e:
        print(f"Error loading routing model: {e}")
        return
    
    # Get encoder/decoder (use baseline checkpoint for this)
    encode, decode = get_encoder_decoder(args.baseline_checkpoint)
    
    # Encode the starting text
    if args.start.startswith('FILE:'):
        with open(args.start[5:], 'r', encoding='utf-8') as f:
            start_text = f.read()
    else:
        start_text = args.start
    
    start_ids = encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    print(f"\nGenerating text with prompt: '{start_text}'")
    print("="*80)
    
    # Generate samples from both models
    for sample_idx in range(args.num_samples):
        print(f"\n{'='*20} SAMPLE {sample_idx + 1} {'='*20}")
        
        # Generate with baseline model
        print(f"\n{'='*15} BASELINE MODEL {'='*15}")
        print(f"Parameters: {sum(p.numel() for p in baseline_model.parameters()):,} (total)")
        baseline_output = generate_text(
            baseline_model, x, args.max_new_tokens, 
            args.temperature, args.top_k, device, ctx
        )
        baseline_text = decode(baseline_output)
        print(baseline_text)
        
        # Generate with routing model
        print(f"\n{'='*15} ROUTING MODEL {'='*15}")
        print(f"Parameters: {sum(p.numel() for p in routing_model.parameters()):,} (total)")
        routing_output = generate_text(
            routing_model, x, args.max_new_tokens, 
            args.temperature, args.top_k, device, ctx
        )
        routing_text = decode(routing_output)
        print(routing_text)
        
        print("\n" + "="*80)

if __name__ == '__main__':
    main() 