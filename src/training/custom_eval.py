import torch
from tqdm import tqdm
from typing import Dict
import gc

@torch.no_grad()
def custom_evaluate(model, eval_dataloader, device, use_bf16=True) -> Dict[str, float]:
    """
    Custom evaluation loop - Tránh OOM
    
    Khác với HF Trainer:
    - KHÔNG lưu logits vào list
    - Giải phóng memory ngay sau mỗi batch
    - Chỉ tính loss, không tính metrics khác
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    print(f"\n{'='*80}")
    print(f"CUSTOM EVALUATION")
    print(f"{'='*80}\n")
    
    for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        try:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }
            
            # Forward pass với autocast
            if use_bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                outputs = model(**batch)
                loss = outputs.loss
            
            # Accumulate loss
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
            
            del outputs, loss, batch
            
            # Clear cache mỗi 5 batches
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️  OOM at batch {batch_idx}, skipping this batch...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e
    
    # Tính average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"  eval_loss: {avg_loss:.4f}")
    print(f"  eval_batches: {num_batches}")
    print(f"{'='*80}\n")
    
    return {
        "eval_loss": avg_loss,
        "eval_samples": num_batches,
        "eval_runtime": 0.0,  # Placeholder
    }