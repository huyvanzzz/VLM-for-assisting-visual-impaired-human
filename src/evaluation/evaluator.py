import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from .metrics import VLMMetrics

class VLMEvaluator:
    def __init__(self, model, tokenizer, processor, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        self.device = config.get('hardware', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.model.to(self.device)
        
        tfidf_path = config.get('evaluation', {}).get('tfidf_path', 'tfidf_vectorizer.pkl')
        self.metrics_engine = VLMMetrics(tfidf_path=tfidf_path)
        
        # Generation configuration
        self.gen_config = {
            "max_new_tokens": 256,
            "num_beams": 3,
            "do_sample": False,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "use_cache": True
        }

    def _split_batch(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """
        Splits input_ids into prompt and reference based on labels (-100 masking).
        """
        prompts_input_ids = []
        references_ids = []
        
        batch_size = input_ids.shape[0]
        
        for i in range(batch_size):
            # Find where the answer starts (first non -100 label)
            valid_label_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
            
            if len(valid_label_indices) == 0:
                # Fallback if no valid labels found
                prompts_input_ids.append(input_ids[i])
                references_ids.append(torch.tensor([], dtype=torch.long))
                continue
                
            start_idx = valid_label_indices[0].item()
            
            # Slice: Prompt is everything before start_idx
            prompts_input_ids.append(input_ids[i][:start_idx])
            
            # Slice: Reference is from start_idx onwards
            references_ids.append(labels[i][valid_label_indices])

        return prompts_input_ids, references_ids

    def generate_batch(self, batch: Dict) -> List[str]:
        """
        Generates text for a batch. Requires batch_size=1 for safety with dynamic images.
        """
        inputs = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
            else:
                inputs[k] = v
                
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        
        # Split prompt from answer
        prompts_list, _ = self._split_batch(input_ids, labels)
        
        decoded_preds = []
        
        with torch.no_grad():
            for i, prompt_ids in enumerate(prompts_list):
                # Prepare single input
                single_input = {
                    'input_ids': prompt_ids.unsqueeze(0).to(self.device),
                    'attention_mask': torch.ones_like(prompt_ids.unsqueeze(0)).to(self.device)
                }
                
                # Handle image inputs (strictly for batch_size=1)
                if 'pixel_values' in inputs:
                    if input_ids.shape[0] == 1:
                        single_input['pixel_values'] = inputs['pixel_values']
                        if 'image_grid_thw' in inputs:
                            single_input['image_grid_thw'] = inputs['image_grid_thw']
                        if 'image_sizes' in inputs:
                            single_input['image_sizes'] = inputs['image_sizes']
                    else:
                        raise ValueError("Evaluation requires batch_size=1 in DataLoader.")

                # Generate
                outputs = self.model.generate(
                    **single_input,
                    **self.gen_config
                )
                
                # Extract only new tokens
                generated_ids = outputs[0][len(prompt_ids):]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                decoded_preds.append(text)
                
        return decoded_preds

    def evaluate_dataset(self, dataloader, task_name: str = "Evaluation", print_samples: int = 5) -> Tuple[Dict, List[str], List[str]]:
        """
        Runs evaluation on the entire dataset.
        Returns: metrics (dict), predictions (list), references (list)
        """
        print("\n" + "="*40)
        print(f"Starting Evaluation: {task_name}")
        print("="*40)
        
        predictions = []
        references = []
        
        print_count = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 1. Generate Prediction
            batch_preds = self.generate_batch(batch)
            predictions.extend(batch_preds)
            
            # 2. Get Ground Truth (Decode from labels)
            _, ref_ids_list = self._split_batch(batch['input_ids'], batch['labels'])
            
            batch_refs = []
            for ref_ids in ref_ids_list:
                # Replace -100 with pad_token_id for safe decoding
                ref_ids = ref_ids.clone()
                ref_ids[ref_ids == -100] = self.tokenizer.pad_token_id
                
                text = self.tokenizer.decode(ref_ids, skip_special_tokens=True)
                batch_refs.append(text)
                
            references.extend(batch_refs)
            
            # 3. Print samples to console for debugging
            if print_count < print_samples:
                print(f"\n[Sample {print_count + 1}]")
                print(f"GT (Ref):  {batch_refs[0]}")
                print(f"Model:     {batch_preds[0]}")
                print("-" * 30)
                print_count += 1

        # 4. Compute Metrics
        scores = self.metrics_engine.compute(
            predictions=predictions,
            references=references,
            target_field="instruction"
        )
        
        print(f"\nResults for {task_name}:")
        for k, v in scores.items():
            print(f"  {k:<10}: {v:.2f}")
            
        return scores, predictions, references