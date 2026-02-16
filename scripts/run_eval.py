import argparse
import yaml
import os
import json
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('.')

# Import project modules
from src.training.trainer import VLMTrainer
from src.data.data_collator import VLMDataCollator
from src.evaluation.evaluator import VLMEvaluator
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run Evaluation for Navigation VLM")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to config.yaml"
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="Path to checkpoint folder. If None, evaluates Base Model."
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="eval_results.json", 
        help="Path to save results JSON"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test_alter",
        choices=["train", "valid", "test_alter", "test_QA"],
        help="Dataset split to evaluate on"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Build model with VLMTrainer
    print("\n" + "="*60)
    if args.checkpoint:
        print("MODE: FINE-TUNED MODEL (LoRA)")
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print("MODE: BASE MODEL (Zero-shot)")
    print("="*60 + "\n")
    
    print("Building model...")
    trainer = VLMTrainer(
        config_path=args.config,
        checkpoint_path=args.checkpoint  # Truyền checkpoint vào đây
    )
    trainer.setup()  # Tự động load checkpoint nếu có
    
    model = trainer.model
    tokenizer = trainer.trainer.tokenizer
    processor = trainer.trainer.processor  # Lấy processor từ trainer

    model.eval()

    # 3. Prepare Dataset
    print(f"Loading dataset split: {args.split}...")
    
    if args.split in ["train", "valid"]:
        # Dùng dataset đã build trong trainer
        if args.split == "train":
            target_dataset = trainer.trainer.train_dataset
        else:  # valid
            target_dataset = trainer.trainer.eval_dataset
            
    else:  # test_alter hoặc test_QA
        from src.data.wad_dataset import WADDataset
        
        if args.split == "test_alter":
            metadata = load_dataset(
                config['data']['name'],
                data_files={"test": "test_alter.json"},
                split="test"
            )
        elif args.split == "test_QA":
            metadata = load_dataset(
                config['data']['name'],
                data_files={"test": "test_QA.json"},
                split="test"
            )
        
        # Build dataset với processor/tokenizer từ trainer
        # Cần expose processor từ trainer
        # Tạm thời: rebuild
        from src.models.model_registry import build_model as get_processor
        vlm_temp = get_processor(config)
        
        target_dataset = WADDataset(
            metadata=metadata,
            processor=vlm_temp.processor,
            tokenizer=vlm_temp.tokenizer,
            config=config,
            num_frames=config['data'].get('num_frames', 1)
        )
    
    print(f"Split: {args.split}")
    print(f"Number of evaluation samples: {len(target_dataset)}")
    
    # 4. Setup DataLoader
    print("Setting up DataLoader (batch_size=1)...")
    data_collator = VLMDataCollator(tokenizer=tokenizer)
    
    eval_dataloader = DataLoader(
        target_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=config['hardware']['num_workers'],
        pin_memory=True
    )

    # 5. Initialize Evaluator
    print("Initializing Evaluator...")
    evaluator = VLMEvaluator(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        config=config
    )

    # 6. Run Evaluation
    print("Starting Evaluation Loop...")
    mode_name = "LoRA_Finetuned" if args.checkpoint else "Base_Model"
    
    metrics, predictions, references = evaluator.evaluate_dataset(
        eval_dataloader, 
        task_name=mode_name,
        print_samples=5
    )

    # 7. Save Detailed Results
    output_path = args.output_file
    print(f"Saving results to {output_path}...")
    
    detailed_samples = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        detailed_samples.append({
            "id": i,
            "ground_truth": ref,
            "prediction": pred,
            "exact_match": pred.strip() == ref.strip()
        })

    final_results = {
        "mode": mode_name,
        "config_file": args.config,
        "checkpoint_path": args.checkpoint,
        "dataset_split": args.split,
        "metrics": metrics,
        "samples": detailed_samples
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
        
    print("\nEVALUATION COMPLETED")
    print(f"  Dataset Split: {args.split}")
    print(f"  Result File: {output_path}")
    print(f"  ROUGE-L: {metrics.get('ROUGE-L', 0):.2f}")
    print(f"  TF-IDF:  {metrics.get('TF-IDF', 0):.2f}")

if __name__ == "__main__":
    main()