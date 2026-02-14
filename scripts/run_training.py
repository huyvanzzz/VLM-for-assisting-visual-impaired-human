#!/usr/bin/env python3
"""
Main training script
Usage:
  - Train from scratch: python scripts/run_training.py --config configs/llava_config.yaml
  - Resume training: python scripts/run_training.py --config configs/llava_config.yaml --resume outputs/qwen_vl/checkpoint-500
  - Load checkpoint và train tiếp: python scripts/run_training.py --config configs/llava_config.yaml --checkpoint outputs/qwen_vl/checkpoint-500
"""

import argparse
import sys
sys.path.append('.')

from src.training.trainer import VLMTrainer
from src.training.utils import set_seed, print_device_info

def main():
    parser = argparse.ArgumentParser(description='Train VLM model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Resume from checkpoint (optimizer + scheduler + steps)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load adapter weights from checkpoint (train from beginning with new optimizer)')
    args = parser.parse_args()
    
    # Print device info
    print_device_info()
    
    trainer = VLMTrainer(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    # Setup (sẽ tự động load checkpoint nếu có)
    trainer.setup()
    
    set_seed(trainer.config['data']['seed'])
    
    # Run training
    if not args.eval_only:
        if args.resume:
            print(f"Resuming full training state from {args.resume}")
            trainer.train(resume_from_checkpoint=args.resume)
        else:
            trainer.train()
    
    # Run evaluation
    results = trainer.evaluate()
    
    # Save model
    if not args.eval_only:
        save_path = trainer.config['training']['output_dir'] + '/final_model'
        trainer.save(save_path)

if __name__ == '__main__':
    main()