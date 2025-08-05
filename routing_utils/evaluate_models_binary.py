#!/usr/bin/env python3
"""
Evaluate routing and baseline models on test data.
Calculates BPC (Bits Per Character) in base 2 for comparison.
"""

import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Import both baseline and routing models
import model
import model_novel

def load_baseline_model(checkpoint_path):
    """Load baseline model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
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
    # Remove routing-specific parameters if they exist
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if not any(routing_key in key for routing_key in ['gate_net', 'routing']):
            filtered_state_dict[key] = value
    
    baseline_model.load_state_dict(filtered_state_dict, strict=False)
    return baseline_model

def load_routing_model(checkpoint_path):
    """Load routing model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Reconstruct model from checkpoint
    model_args = checkpoint['model_args']
    routing_model = model_novel.GPT(model_novel.GPTConfig(**model_args))
    
    # Load state dict
    routing_model.load_state_dict(checkpoint['model'])
    return routing_model

def get_bpc_from_loss(loss_nats):
    """Convert loss from nats to BPC (base 2)."""
    return loss_nats / math.log(2)

def load_binary_data(data_path):
    """Load binary data using memmap (same as train.py)."""
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    return data

def load_text_data(data_path):
    """Load text data (for non-binary datasets)."""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def evaluate_model(model, data, block_size, batch_size, device, use_full_dataset=False):
    """Evaluate model on test data."""
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        if use_full_dataset:
            # Evaluate on the entire test dataset
            print("Evaluating on entire test dataset...")
            num_samples = len(data) - block_size
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                actual_batch_size = end_idx - start_idx
                
                # Create sequences
                x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in range(start_idx, end_idx)])
                y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in range(start_idx, end_idx)])
                
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                output = model(x, y)
                
                # Handle tuple output from routing model
                if isinstance(output, tuple):
                    logits, loss = output
                else:
                    logits = output
                    # For baseline model, we need to compute loss manually
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                
                # Accumulate loss and token count
                total_loss += loss.item() * actual_batch_size * block_size
                total_tokens += actual_batch_size * block_size
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"Processed {batch_idx + 1}/{num_batches} batches...")
            
            # Average loss across all tokens
            avg_loss = total_loss / total_tokens
        else:
            # Sample random batches (original method)
            print("Evaluating on sampled batches...")
            num_batches = 100
            total_batches = 0
            
            for _ in range(num_batches):
                # Sample random starting points
                ix = torch.randint(len(data) - block_size, (batch_size,))
                x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
                y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
                
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                output = model(x, y)
                
                # Handle tuple output from routing model
                if isinstance(output, tuple):
                    logits, loss = output
                else:
                    logits = output
                    # For baseline model, we need to compute loss manually
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                
                total_loss += loss.item()
                total_batches += 1
            
            # Average loss across all batches
            avg_loss = total_loss / total_batches
    
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Evaluate routing and baseline models')
    parser.add_argument('--routing_checkpoint', type=str, 
                       default='out-enwik8-routing/ckpt.pt',
                       help='Path to routing model checkpoint')
    parser.add_argument('--baseline_checkpoint', type=str,
                       default='out-enwik8-routing-ablation/ckpt.pt',
                       help='Path to baseline model checkpoint')
    parser.add_argument('--test_data', type=str,
                       default='data/enwik8/test.bin',
                       help='Path to test data')
    parser.add_argument('--data_type', type=str, choices=['auto', 'binary', 'text'],
                       default='auto', help='Type of data file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_batches', type=int, default=100,
                       help='Number of batches to evaluate (when not using full dataset)')
    parser.add_argument('--use_full_dataset', action='store_true',
                       help='Evaluate on entire test dataset instead of sampling')
    parser.add_argument('--output_file', type=str,
                       help='Path to save results JSON')
    
    args = parser.parse_args()
    
    # Auto-detect data type
    if args.data_type == 'auto':
        if args.test_data.endswith('.bin'):
            args.data_type = 'binary'
        else:
            args.data_type = 'text'
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    if args.data_type == 'binary':
        test_data = load_binary_data(args.test_data)
    else:
        test_data = load_text_data(args.test_data)
    
    print(f"Test data loaded: {len(test_data)} tokens")
    
    # Load models
    print("Loading models...")
    try:
        routing_model = load_routing_model(args.routing_checkpoint)
        print(f"Routing model loaded from {args.routing_checkpoint}")
        print(f"Routing model parameters: {sum(p.numel() for p in routing_model.parameters()):,} (total)")
        print(f"Routing model parameters (non-embedding): {routing_model.get_num_params():,}")
    except Exception as e:
        print(f"Error loading routing model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        baseline_model = load_baseline_model(args.baseline_checkpoint)
        print(f"Baseline model loaded from {args.baseline_checkpoint}")
        print(f"Baseline model parameters: {sum(p.numel() for p in baseline_model.parameters()):,} (total)")
        print(f"Baseline model parameters (non-embedding): {baseline_model.get_num_params():,}")
    except Exception as e:
        print(f"Error loading baseline model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get model configurations
    routing_config = routing_model.config
    baseline_config = baseline_model.config
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Evaluate models
    print("\nEvaluating models...")
    
    # Evaluate routing model
    print("Evaluating routing model...")
    routing_loss = evaluate_model(
        routing_model, test_data, 
        routing_config.block_size, args.batch_size, 
        device, use_full_dataset=args.use_full_dataset
    )
    routing_bpc = get_bpc_from_loss(routing_loss)
    print(f"Routing model evaluation complete: Loss = {routing_loss:.6f} nats, BPC = {routing_bpc:.6f}")
    
    # Evaluate baseline model
    print("Evaluating baseline model...")
    baseline_loss = evaluate_model(
        baseline_model, test_data,
        baseline_config.block_size, args.batch_size,
        device, use_full_dataset=args.use_full_dataset
    )
    baseline_bpc = get_bpc_from_loss(baseline_loss)
    print(f"Baseline model evaluation complete: Loss = {baseline_loss:.6f} nats, BPC = {baseline_bpc:.6f}")
    
    # Calculate improvement
    bpc_improvement = baseline_bpc - routing_bpc
    improvement_percentage = (bpc_improvement / baseline_bpc) * 100
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Routing Model:")
    print(f"  Loss: {routing_loss:.6f} nats")
    print(f"  BPC:  {routing_bpc:.6f} bits/char")
    print(f"  Parameters: {sum(p.numel() for p in routing_model.parameters()):,} (total)")
    print(f"  Parameters: {routing_model.get_num_params():,} (non-embedding)")
    
    print(f"\nBaseline Model:")
    print(f"  Loss: {baseline_loss:.6f} nats")
    print(f"  BPC:  {baseline_bpc:.6f} bits/char")
    print(f"  Parameters: {sum(p.numel() for p in baseline_model.parameters()):,} (total)")
    print(f"  Parameters: {baseline_model.get_num_params():,} (non-embedding)")
    
    print(f"\nComparison:")
    print(f"  BPC Improvement: {bpc_improvement:.6f} bits/char")
    print(f"  Improvement: {improvement_percentage:.2f}%")
    print(f"  Parameter Increase (total): {sum(p.numel() for p in routing_model.parameters()) - sum(p.numel() for p in baseline_model.parameters()):,}")
    print(f"  Parameter Increase (non-embedding): {routing_model.get_num_params() - baseline_model.get_num_params():,}")
    
    # Save results
    results = {
        'routing_model': {
            'loss_nats': float(routing_loss),
            'bpc': float(routing_bpc),
            'parameters_total': int(sum(p.numel() for p in routing_model.parameters())),
            'parameters_non_embedding': int(routing_model.get_num_params())
        },
        'baseline_model': {
            'loss_nats': float(baseline_loss),
            'bpc': float(baseline_bpc),
            'parameters_total': int(sum(p.numel() for p in baseline_model.parameters())),
            'parameters_non_embedding': int(baseline_model.get_num_params())
        },
        'comparison': {
            'bpc_improvement': float(bpc_improvement),
            'improvement_percentage': float(improvement_percentage),
            'parameter_increase_total': int(sum(p.numel() for p in routing_model.parameters()) - sum(p.numel() for p in baseline_model.parameters())),
            'parameter_increase_non_embedding': int(routing_model.get_num_params() - baseline_model.get_num_params()),
            'routing_better': bool(bpc_improvement > 0)
        },
        'evaluation_settings': {
            'test_data_path': args.test_data,
            'data_type': args.data_type,
            'batch_size': args.batch_size,
            'use_full_dataset': args.use_full_dataset,
            'num_batches': args.num_batches if not args.use_full_dataset else None,
            'device': device
        }
    }
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")
    else:
        # Save to default location
        output_file = 'evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main() 