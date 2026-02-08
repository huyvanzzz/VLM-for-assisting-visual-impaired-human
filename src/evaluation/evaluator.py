import torch
from typing import Dict, List
from tqdm import tqdm
import json

class VLMEvaluator:
    """High-level evaluation orchestration"""
    
    def __init__(self, model, tokenizer, processor, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.device = config['hardware']['device']
    
    def generate_predictions(self, eval_dataset) -> List[str]:
        """Generate predictions for eval dataset"""
        
        self.model.eval()
        predictions = []
        
        print("\n" + "="*80)
        print("GENERATING PREDICTIONS")
        print("="*80 + "\n")
        
        with torch.no_grad():
            for idx in tqdm(range(len(eval_dataset)), desc="Evaluating"):
                sample = eval_dataset[idx]
                
                # Prepare inputs
                inputs = {
                    'input_ids': sample['input_ids'].unsqueeze(0).to(self.device),
                    'attention_mask': sample['attention_mask'].unsqueeze(0).to(self.device),
                    'pixel_values': sample['pixel_values'].unsqueeze(0).to(self.device)
                }
                
                # Optional keys
                if 'image_sizes' in sample:
                    inputs['image_sizes'] = [tuple(sample['image_sizes'].tolist())]
                
                if 'image_grid_thw' in sample:
                    grid = sample['image_grid_thw']
                    if grid.dim() == 1:
                        grid = grid.unsqueeze(0)
                    inputs['image_grid_thw'] = grid.unsqueeze(0).to(self.device)
                
                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1
                )
                
                # Decode
                pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract JSON part
                try:
                    start_idx = pred_text.find('{')
                    end_idx = pred_text.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_text = pred_text[start_idx:end_idx]
                        json.loads(json_text)  # Validate
                        predictions.append(json_text)
                    else:
                        predictions.append("{}")
                except:
                    predictions.append("{}")
        
        print(f"\n✓ Generated {len(predictions)} predictions")
        
        return predictions
    
    def evaluate(self, eval_dataset) -> Dict[str, float]:
        """Full evaluation pipeline"""
        
        from .metrics import compute_metrics
        
        # Generate predictions
        predictions = self.generate_predictions(eval_dataset)
        
        # Extract references
        references = []
        for idx in range(len(eval_dataset)):
            sample = eval_dataset[idx]
            labels = sample['labels']
            gt_tokens = labels[labels != -100]
            ref_text = self.tokenizer.decode(gt_tokens, skip_special_tokens=True)
            references.append(ref_text)
        
        # Compute metrics
        results = compute_metrics(predictions, references)
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        for metric, score in results.items():
            print(f"  {metric}: {score:.2f}%")
        
        print("="*80 + "\n")
        
        return results