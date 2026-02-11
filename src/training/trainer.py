from transformers import Trainer, TrainingArguments
from typing import Dict, Any
import yaml
import torch
from tqdm.auto import tqdm
import gc
import os

class VLMTrainer:
    """High-level training orchestration"""
    
    def __init__(self, config_path: str):
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.trainer = None
        self.results = {}
        self.pbar = None  # Progress bar
    
    def setup(self):
        """Setup model, data, trainer"""
        self._clear_memory()

        from ..models.model_registry import build_model
        from ..data.wad_dataset import build_dataset
        from .callbacks import MemoryOptimizationCallback, ExperimentTrackingCallback
        from ..data.data_collator import VLMDataCollator
        
        # Tạo progress bar cho setup
        setup_steps = ["Building model", "Building dataset", "Creating trainer"]
        with tqdm(total=len(setup_steps), desc="Setup Progress") as pbar:
            # Build model
            pbar.set_description("Building model...")
            vlm = build_model(self.config)
            self.model = vlm.model
            pbar.update(1)
            
            # Build dataset
            pbar.set_description("Building dataset...")
            train_dataset, eval_dataset = build_dataset(
                self.config,
                vlm.processor,
                vlm.tokenizer
            )
            pbar.update(1)
            
            # Training arguments
            pbar.set_description("Creating trainer...")
            training_args = TrainingArguments(
                output_dir=self.config['training']['output_dir'],
                num_train_epochs=self.config['training']['num_epochs'],
                per_device_train_batch_size=self.config['training']['batch_size'],
                gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
                learning_rate=float(self.config['training']['learning_rate']),
                warmup_steps=int(self.config['training']['warmup_steps']),
                weight_decay=float(self.config['training']['weight_decay']),
                fp16=self.config['training']['fp16'],
                gradient_checkpointing=self.config['training']['gradient_checkpointing'],
                logging_steps=self.config['training']['logging_steps'],
                eval_steps=self.config['training']['eval_steps'],
                save_steps=self.config['training']['save_steps'],
                save_total_limit=self.config['training']['save_total_limit'],
                remove_unused_columns=False,
                dataloader_pin_memory=self.config['hardware']['pin_memory'],
                dataloader_num_workers=self.config['hardware']['num_workers'],
                report_to="none",
                optim=self.config['training']['optimizer'],
                disable_tqdm=False,  # Bật tqdm của Trainer
            )
            
            data_collator = VLMDataCollator(tokenizer=vlm.tokenizer)
                
            # Callbacks
            callbacks = [
                MemoryOptimizationCallback(
                    clear_cache_steps=25,  # Có thể giảm xuống 10 nếu vẫn OOM
                    log_memory_steps=10
                ),
            ]
            
            if self.config['tracking']['enabled']:
                callbacks.append(ExperimentTrackingCallback(self.config))
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=vlm.tokenizer,
                data_collator=data_collator,
                callbacks=callbacks
            )
            pbar.update(1)
        
        print("✓ Setup complete!")
    
    def train(self):
        """Run training"""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")
        
        self._clear_memory() 
        # Trainer đã có tqdm built-in, chỉ cần đảm bảo disable_tqdm=False
        self.trainer.train()
        
        print("\n✓ Training complete!")
        self._clear_memory()

    def evaluate(self):
        """Run evaluation"""
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80 + "\n")
        self._clear_memory()
        results = self.trainer.evaluate()
        self.results = results
        
        print(results)
        self._clear_memory() 
        return results
    
    def save(self, output_path: str):

        self._clear_memory()
        """Save model"""
        print(f"\nSaving model to {output_path}...")
        with tqdm(total=1, desc="Saving model") as pbar:
            self.trainer.save_model(output_path)
            pbar.update(1)
        print(f"✓ Model saved to {output_path}")
        self._clear_memory()