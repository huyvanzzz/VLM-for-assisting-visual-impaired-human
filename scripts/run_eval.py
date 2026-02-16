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
        default="test_alter",
        choices=["train", "valid", "test_alter", "test_QA"],  # Giữ "valid"
        help="Dataset split to evaluate on"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # ✅ FIX: Tắt LoRA trong config khi có checkpoint
    # Vì checkpoint đã chứa LoRA weights rồi
    if args.checkpoint:
        print("\n⚠️  Disabling LoRA in config (will load from checkpoint)")
        if 'model' in config and 'lora' in config['model']:
            config['model']['lora']['enabled'] = False
    
    # 2. Build Base Model (không có LoRA nếu có checkpoint)
    print("Building Base Model...")
    vlm_wrapper = build_model(config)
    
    model = vlm_wrapper.model
    tokenizer = vlm_wrapper.tokenizer
    processor = vlm_wrapper.processor
    
    # 3. Load LoRA từ checkpoint
    if args.checkpoint:
        print("\n" + "="*60)
        print("MODE: FINE-TUNED MODEL (LoRA)")
        print(f"Checkpoint: {args.checkpoint}")
        print("="*60 + "\n")
        
        if not os.path.exists(args.checkpoint):
            raise ValueError(f"Checkpoint not found: {args.checkpoint}")
        
        # Validate files
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        missing_files = [f for f in required_files 
                        if not os.path.exists(os.path.join(args.checkpoint, f))]
        
        if missing_files:
            # Check for .bin as fallback
            if 'adapter_model.safetensors' in missing_files:
                if os.path.exists(os.path.join(args.checkpoint, 'adapter_model.bin')):
                    missing_files.remove('adapter_model.safetensors')
            
            if missing_files:
                raise ValueError(f"Missing files: {missing_files}")
        
        print("Loading LoRA adapter...")
        try:
            model = PeftModel.from_pretrained(
                model,
                args.checkpoint,
                torch_dtype=torch.bfloat16 if config['training'].get('bf16', False) 
                           else torch.float16 if config['training']['fp16'] 
                           else torch.float32,
                is_trainable=False  # Evaluation mode
            )
            print("✓ LoRA Adapter loaded successfully.")
            
            # Print adapter info
            if hasattr(model, 'print_trainable_parameters'):
                model.print_trainable_parameters()
                
        except Exception as e:
            print(f"❌ Error loading adapter: {e}")
            raise
            
    else:
        print("\n" + "="*60)
        print("MODE: BASE MODEL (Zero-shot)")
        print("="*60 + "\n")
    
    # Set device and eval mode
    device = config.get('hardware', {}).get('device', 
             'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"✓ Model ready on device: {device}\n")

    # 4. Prepare Dataset
    print(f"Loading dataset split: {args.split}...")
    
    if args.split in ["train", "valid"]:
        # Dùng build_dataset cho train/valid (tự động chia)
        train_dataset, valid_dataset = build_dataset(config, processor, tokenizer)
        
        if args.split == "train":
            target_dataset = train_dataset
        else:  # valid
            target_dataset = valid_dataset
            
    else:  # test_alter hoặc test_QA
        # Load test splits riêng
        from datasets import load_dataset
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
        
        target_dataset = WADDataset(
            metadata=metadata,
            processor=processor,
            tokenizer=tokenizer,
            config=config,
            num_frames=config['data'].get('num_frames', 1)
        )
    
    print(f"Split: {args.split}")
    print(f"Number of evaluation samples: {len(target_dataset)}")
    
    # 5. Setup DataLoader
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
    
    metrics, predictions, references = evaluator.evaluate_dataset(
        eval_dataloader, 
        task_name=mode_name,
        print_samples=5
    )

    # 8. Save Detailed Results
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