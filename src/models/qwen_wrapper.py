from .base_vlm import BaseVLM
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

class QwenVLModel(BaseVLM):
    """Qwen-VL model wrapper"""
    
    def load_model(self):
        # Similar to LLaVA but for Qwen-VL architecture
        
        if self.config['model']['quantization']['enabled']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config['model']['quantization']['type'],
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=self.config['model']['quantization']['double_quant']
            )
        else:
            bnb_config = None
        
        config = AutoConfig.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True
        )
        config.use_cache = False
        
        # Qwen-VL uses AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            quantization_config=bnb_config,
            config=config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16
        )
        
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config['training']['gradient_checkpointing']
        )
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        return self.model
    
    def load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True,
            use_fast=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.processor, self.tokenizer
    
    def apply_lora(self, lora_config):
        lora_cfg = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def freeze_vision_encoder(self):
        frozen_count = 0
        vision_keywords = ['visual', 'transformer', 'vit']
        skip_keywords = ['adapter', 'projector']
        
        for name, param in self.model.named_parameters():
            if any(kw in name.lower() for kw in vision_keywords):
                if not any(skip in name.lower() for skip in skip_keywords):
                    param.requires_grad = False
                    frozen_count += 1
        
        print(f"✓ Frozen {frozen_count} vision parameters")
    
    def prepare_inputs(self, batch):
        return batch