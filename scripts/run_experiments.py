#!/usr/bin/env python3
"""
Run multiple experiments
Usage: python scripts/run_experiments.py --configs configs/llava_config.yaml configs/qwen_config.yaml
"""

import argparse
import sys
sys.path.append('.')

from src.training.trainer import VLMTrainer
from src.utils.visualization import plot_model_comparison
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run multiple VLM experiments')
    parser.add_argument('--configs', nargs='+', required=True, help='List of config files')
    parser.add_argument('--output', type=str, default='./experiments/comparison.json', help='Output file')
    args = parser.parse_args()
    
    # Create output dir
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for config_path in args.configs:
        print(f"\n{'='*80}")
        print(f"Running experiment: {config_path}")
        print(f"{'='*80}\n")
        
        try:
            trainer = VLMTrainer(config_path)
            trainer.setup()
            trainer.train()
            results = trainer.evaluate()
            
            model_name = trainer.config['experiment']['name']
            all_results[model_name] = results
            
            print(f"\n✓ {model_name} completed")
            
        except Exception as e:
            print(f"\n✗ {config_path} failed: {e}")
            continue
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"\n✓ Results saved to {args.output}")
    
    # Plot comparison
    if len(all_results) > 1:
        plot_model_comparison(all_results, args.output.replace('.json', '.png'))
    
    # Print summary
    print("\nSummary:")
    for model, results in all_results.items():
        print(f"\n{model}:")
        for metric, score in results.items():
            print(f"  {metric}: {score:.2f}%")

if __name__ == '__main__':
    main()