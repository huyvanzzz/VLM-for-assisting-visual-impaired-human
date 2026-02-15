from transformers import TrainerCallback
import torch
import gc

class CustomEvalCallback(TrainerCallback):
    """
    Callback để chạy custom evaluation trong quá trình training
    """
    
    def __init__(self, vlm_trainer):
        self.vlm_trainer = vlm_trainer
    
    def on_evaluate(self, args, state, control, **kwargs):
        """
        Gọi sau khi HF Trainer chạy evaluate()
        Nhưng ta sẽ override bằng custom eval
        """
        # Clear memory trước khi eval
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\n🔄 Running custom evaluation...")
        
        # Gọi custom eval
        results = self.vlm_trainer.evaluate()
        
        # Log results
        if state.is_world_process_zero:
            print(f"\n📊 Custom Eval Results:")
            print(f"  Step {state.global_step}: eval_loss = {results['eval_loss']:.4f}")
        
        # Clear memory sau eval
        torch.cuda.empty_cache()
        gc.collect()
        
        return control