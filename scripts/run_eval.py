import argparse
import yaml
import os
import json
import torch
from torch.utils.data import DataLoader
from peft import PeftModel
import sys
sys.path.append('.')
# Import project modules
from src.models.model_registry import build_model
from src.data.wad_dataset import build_dataset
from src.data.data_collator import VLMDataCollator
from src.evaluation.evaluator import VLMEvaluator

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
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate on"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Build Base Model
    print("Building Base Model...")
    vlm_wrapper = build_model(config)
    
    model = vlm_wrapper.model
    tokenizer = vlm_wrapper.tokenizer
    processor = vlm_wrapper.processor
    
    # 3. Handle Base vs LoRA Logic
    if args.checkpoint:
        print("\n" + "="*60)
        print("MODE: FINE-TUNED MODEL (LoRA)")
        print(f"Checkpoint: {args.checkpoint}")
        print("="*60 + "\n")
        
        if os.path.exists(args.checkpoint):
            model = PeftModel.from_pretrained(
                model,
                args.checkpoint,
                torch_dtype=torch.float16 if config['training']['fp16'] else torch.float32
            )
            print("LoRA Adapter loaded successfully.")
        else:
            raise ValueError(f"Checkpoint path not found: {args.checkpoint}")
    else:
        print("\n" + "="*60)
        print("MODE: BASE MODEL (Zero-shot)")
        print("Warning: Base model might output raw text formats.")
        print("="*60 + "\n")

    model.eval()

    # 4. Prepare Dataset
    print("Loading dataset...")
    train_subset, val_subset = build_dataset(
        config, 
        processor, 
        tokenizer
    )
    
    target_dataset = train_subset if args.split == "train" else val_subset
    print(f"Number of evaluation samples: {len(target_dataset)}")

    # 5. Setup DataLoader
    print("Setting up DataLoader (batch_size=1)...")
    data_collator = VLMDataCollator(tokenizer=tokenizer)
    
    eval_dataloader = DataLoader(
        target_dataset,
        batch_size=1,            # REQUIRED: batch_size=1
        shuffle=False,
        collate_fn=data_collator,
        num_workers=config['hardware']['num_workers'],
        pin_memory=True
    )

    # 6. Initialize Evaluator
    print("Initializing Evaluator...")
    evaluator = VLMEvaluator(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        config=config
    )

    # 7. Run Evaluation
    print("Starting Evaluation Loop...")
    mode_name = "LoRA_Finetuned" if args.checkpoint else "Base_Model"
    
    # Capture metrics, predictions, and references
    metrics, predictions, references = evaluator.evaluate_dataset(
        eval_dataloader, 
        task_name=mode_name,
        print_samples=5  # Print 5 samples to console
    )

    # 8. Save Detailed Results
    output_path = args.output_file
    print(f"Saving results to {output_path}...")
    
    # Create detailed list of samples
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
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
        
    print("\nEVALUATION COMPLETED")
    print(f"  Result File: {output_path}")
    print(f"  ROUGE-L: {metrics.get('ROUGE-L', 0):.2f}")
    print(f"  TF-IDF:  {metrics.get('TF-IDF', 0):.2f}")

if __name__ == "__main__":
    main()