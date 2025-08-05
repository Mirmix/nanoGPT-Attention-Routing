#!/usr/bin/env python3
"""
Plot training curves from training_log.json files.
Supports plotting individual model curves or comparing multiple models.
"""

import json
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_log(log_file):
    """Load training log from JSON file."""
    with open(log_file, 'r') as f:
        return json.load(f)

def plot_single_model(log_file, save_path=None):
    """Plot training curves for a single model."""
    data = load_training_log(log_file)
    
    # Extract data
    iterations = data['iterations']
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    learning_rates = data['learning_rates']
    
    # Convert losses to BPC (base 2)
    train_bpc = [loss / math.log(2) for loss in train_losses]
    val_bpc = [loss / math.log(2) for loss in val_losses]
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Training and Validation BPC
    ax1.plot(iterations, train_bpc, label='Training BPC', color='blue', alpha=0.7)
    ax1.plot(iterations, val_bpc, label='Validation BPC', color='red', alpha=0.7)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Bits Per Character (Base 2)')
    ax1.set_title('Training and Validation BPC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning Rate
    # Skip the initial 0 value for better visualization
    non_zero_lr_indices = [i for i, lr in enumerate(learning_rates) if lr > 0]
    if non_zero_lr_indices:
        start_idx = non_zero_lr_indices[0]
        ax2.plot(iterations[start_idx:], learning_rates[start_idx:], color='green')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: BPC Difference (Overfitting indicator)
    bpc_diff = [t - v for t, v in zip(train_bpc, val_bpc)]
    ax3.plot(iterations, bpc_diff, color='purple', alpha=0.7)
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('BPC Difference (Train - Val)')
    ax3.set_title('Overfitting Indicator')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print(f"\nTraining Summary for {log_file}:")
    print(f"Final Training BPC: {train_bpc[-1]:.6f}")
    print(f"Final Validation BPC: {val_bpc[-1]:.6f}")
    print(f"Best Validation BPC: {min(val_bpc):.6f}")
    print(f"Training Iterations: {len(iterations)}")

def plot_comparison(log_files, model_names, save_path=None):
    """Plot comparison of multiple models."""
    if len(log_files) != len(model_names):
        raise ValueError("Number of log files must match number of model names")
    
    # Load all data
    all_data = []
    for log_file in log_files:
        all_data.append(load_training_log(log_file))
    
    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (data, name, color) in enumerate(zip(all_data, model_names, colors)):
        iterations = data['iterations']
        val_losses = data['val_losses']
        val_bpc = [loss / math.log(2) for loss in val_losses]
        
        ax.plot(iterations, val_bpc, label=name, color=color, alpha=0.8)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Validation BPC (Base 2)')
    ax.set_title('Model Comparison: Validation BPC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    # Print comparison summary
    print("\nModel Comparison Summary:")
    for i, (data, name) in enumerate(zip(all_data, model_names)):
        val_losses = data['val_losses']
        val_bpc = [loss / math.log(2) for loss in val_losses]
        print(f"{name}: Final Val BPC = {val_bpc[-1]:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from JSON log files')
    parser.add_argument('--log_file', type=str, help='Path to training_log.json file')
    parser.add_argument('--log_files', nargs='+', help='Multiple log files for comparison')
    parser.add_argument('--model_names', nargs='+', help='Names for models in comparison')
    parser.add_argument('--save_path', type=str, help='Path to save the plot')
    parser.add_argument('--comparison_save_path', type=str, help='Path to save comparison plot')
    
    args = parser.parse_args()
    
    if args.log_file:
        # Single model plotting
        plot_single_model(args.log_file, args.save_path)
    elif args.log_files and args.model_names:
        # Comparison plotting
        if len(args.log_files) != len(args.model_names):
            print("Error: Number of log files must match number of model names")
            return
        plot_comparison(args.log_files, args.model_names, args.comparison_save_path)
    else:
        # Auto-detect and compare both models
        routing_log = 'out-enwik8-routing/training_log.json'
        baseline_log = 'out-enwik8-routing-ablation/training_log.json'
        
        # Check if both files exist
        if Path(routing_log).exists() and Path(baseline_log).exists():
            print("Auto-detected both training logs. Generating comparison...")
            plot_comparison([routing_log, baseline_log], ['Routing', 'Baseline'], 'training_plots/model_comparison.png')
        elif Path(routing_log).exists():
            print("Found routing model log. Plotting single model...")
            plot_single_model(routing_log, 'training_plots/routing_curves.png')
        elif Path(baseline_log).exists():
            print("Found baseline model log. Plotting single model...")
            plot_single_model(baseline_log, 'training_plots/baseline_curves.png')
        else:
            print("No training logs found. Please provide arguments:")
            print("  python plot_training_curves.py --log_file out-enwik8-routing/training_log.json")
            print("  python plot_training_curves.py --log_files out-enwik8-routing/training_log.json out-enwik8-routing-ablation/training_log.json --model_names Routing Baseline")
            print("\nOr ensure training logs exist at:")
            print(f"  {routing_log}")
            print(f"  {baseline_log}")

if __name__ == '__main__':
    main() 