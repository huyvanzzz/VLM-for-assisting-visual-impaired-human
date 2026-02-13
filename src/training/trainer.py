from transformers import Trainer, TrainingArguments
from typing import Dict, Any
import yaml
import torch
from tqdm.auto import tqdm
import gc
import os
import warnings

class VLMTrainer:
    """High-level training orchestration"""
    
    def __init__(self, config_path: str):
        warnings.filterwarnings('ignore', message='.*Unused or unrecognized kwargs.*')

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
            print("\n" + "="*40)
            print("🩺 [SANITY CHECK] Kiểm tra cấu hình Model")
            try:
                # 1. Check Config (Chỉ in thông số quan trọng, không in cả cụm để tránh lỗi JSON)
                if hasattr(self.model.config, "vision_config"):
                    vc = self.model.config.vision_config
                    # Lấy thông số an toàn
                    hidden_size = getattr(vc, 'hidden_size', 'N/A')
                    patch_size = getattr(vc, 'patch_size', 14) # Mặc định 14 nếu ko tìm thấy
                    image_size = getattr(vc, 'image_size', 336) # Mặc định 336
                    
                    print(f" - Vision Params: Size={image_size}, Patch={patch_size}, Hidden={hidden_size}")
                else:
                    print(" - Không tìm thấy vision_config (Model lạ?)")
                    vc = None

                # 2. Check Thực tế (Chạy thử 1 ảnh rỗng)
                print(" - Đang chạy thử Vision Tower để đếm token...")
                
                # Tìm Vision Tower
                tower = None
                if hasattr(self.model, "vision_tower"):
                    tower = self.model.vision_tower
                elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_tower"):
                    tower = self.model.model.vision_tower
                
                if tower is not None:
                    # Tạo ảnh giả đúng device/dtype của model
                    device = self.model.device
                    dtype = self.model.dtype if self.model.dtype not in [torch.float32, torch.float16, torch.bfloat16] else torch.float16 # Fallback an toàn
                    
                    # Lấy image size thực tế
                    img_s = image_size if isinstance(image_size, int) else 336
                    
                    # Tạo input
                    dummy_pixel = torch.zeros(1, 3, img_s, img_s).to(device)
                    # Cast về đúng kiểu dữ liệu model đang dùng
                    dummy_pixel = dummy_pixel.to(dtype=self.model.dtype)

                    with torch.no_grad():
                        # Một số model output ra tuple, một số ra tensor luôn
                        outputs = tower(dummy_pixel)
                        
                        # Xử lý output để lấy features cuối cùng
                        if isinstance(outputs, (tuple, list)):
                            features = outputs[-1] # Thường hidden states nằm cuối hoặc đầu
                            # Nếu vẫn là tuple (hidden_states), lấy cái cuối
                            if isinstance(features, (tuple, list)):
                                features = features[-1]
                        elif hasattr(outputs, "last_hidden_state"):
                             features = outputs.last_hidden_state
                        else:
                            features = outputs

                    # Đếm số token
                    # Features shape: [Batch, Num_Tokens, Hidden]
                    real_tokens = features.shape[1]
                    
                    # Tính toán lý thuyết
                    grid_w = img_s // patch_size
                    expected_tokens = grid_w * grid_w
                    
                    print(f"   + Lý thuyết (Grid {grid_w}x{grid_w}): {expected_tokens}")
                    print(f"   + Thực tế Vision trả về: {real_tokens}")
                    
                    diff = real_tokens - expected_tokens
                    if diff == 1:
                        print(f" -> 🚨 KẾT LUẬN: Model có thêm 1 token (CLS/Global). CODE FIX CẦN ĐƯỢC BẬT!")
                    elif diff == 0:
                        print(f" -> ✅ KẾT LUẬN: Số lượng khớp (Không có CLS).")
                    else:
                         print(f" -> ⚠️ Lệch {diff} token (Có thể do kiến trúc đặc biệt).")
                else:
                    print(" - Không tìm thấy module vision_tower để test.")

            except Exception as e:
                print(f" - (Check thất bại do lỗi code check: {e})")
                # In traceback để debug nếu cần
                # import traceback
                # traceback.print_exc()
            print("="*40 + "\n")
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
    def _clear_memory(self):
        """Clear GPU and CPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()