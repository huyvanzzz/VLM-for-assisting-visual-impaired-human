"""
Q-VLM Quantization Module
Ported from original notebook for optional use
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy
from typing import List
from tqdm import tqdm
import time


class VisionEncoderOptimizer:
    """Optimize Vision Encoder (VEO step)"""
    
    def __init__(self, vision_encoder: nn.Module, device: torch.device):
        self.vision_encoder = vision_encoder
        self.device = device
        
    def compute_entropy(self, activations: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(activations.flatten(), dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return entropy
    
    def loss_regression(self, original_output: torch.Tensor, optimized_output: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(optimized_output, original_output)
    
    def loss_entropy(self, activations: torch.Tensor) -> torch.Tensor:
        return self.compute_entropy(activations)
    
    def loss_quantization(self, activations: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
        scale = (2 ** num_bits - 1) / (activations.max() - activations.min() + 1e-10)
        quantized = torch.round((activations - activations.min()) * scale) / scale + activations.min()
        return F.mse_loss(quantized, activations)
    
    def optimize(
        self,
        calibration_loader: DataLoader,
        lambda_reg: float = 1.0,
        lambda_entropy: float = 0.1,
        lambda_quant: float = 0.5,
        num_iterations: int = 30
    ):
        """Run VEO optimization"""
        
        original_encoder = copy.deepcopy(self.vision_encoder)
        original_encoder.eval()
        for param in original_encoder.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.Adam(self.vision_encoder.parameters(), lr=1e-5)
        self.vision_encoder.train()
        
        print(f"  VEO: {num_iterations} iterations")
        
        for iteration in tqdm(range(num_iterations), desc="VEO"):
            for batch in calibration_loader:
                try:
                    optimizer.zero_grad()
                    
                    pixel_values = batch['pixel_values'].to(self.device)
                    
                    # Handle image_grid_thw if exists
                    if 'image_grid_thw' in batch:
                        image_grid_thw = batch['image_grid_thw'].to(self.device)
                        if image_grid_thw.dim() == 3:
                            image_grid_thw = image_grid_thw.squeeze(0)
                        
                        with torch.no_grad():
                            original_output = original_encoder(pixel_values, grid_thw=image_grid_thw)
                        
                        optimized_output = self.vision_encoder(pixel_values, grid_thw=image_grid_thw)
                    else:
                        with torch.no_grad():
                            original_output = original_encoder(pixel_values)
                        
                        optimized_output = self.vision_encoder(pixel_values)
                    
                    # Compute losses
                    l_reg = self.loss_regression(original_output, optimized_output)
                    l_entropy = self.loss_entropy(optimized_output)
                    l_quant = self.loss_quantization(optimized_output)
                    
                    loss = lambda_reg * l_reg + lambda_entropy * l_entropy + lambda_quant * l_quant
                    
                    loss.backward()
                    optimizer.step()
                    
                except Exception as e:
                    continue
        
        print(f"  ✓ VEO completed")
        return self.vision_encoder


class ActivationQuantizer(nn.Module):
    """W4A4 activation quantizer"""
    
    def __init__(self, num_bits=4):
        super().__init__()
        self.num_bits = num_bits
        self.qmax = 2 ** num_bits - 1
        
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.calibrated = False
    
    def calibrate(self, x: torch.Tensor):
        x_min = x.min().item()
        x_max = x.max().item()
        
        if x_max - x_min < 1e-8:
            self.scale = torch.tensor(1.0)
            self.zero_point = torch.tensor(0.0)
        else:
            self.scale = torch.tensor((x_max - x_min) / self.qmax)
            self.zero_point = torch.tensor(-x_min / self.scale)
        
        self.calibrated = True
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.calibrated:
            self.calibrate(x)
        
        x_q = torch.clamp(
            torch.round(x / self.scale + self.zero_point),
            0, self.qmax
        )
        
        x_dq = (x_q - self.zero_point) * self.scale
        return x_dq
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantize(x)


def insert_activation_quantizers(model: nn.Module) -> nn.Module:
    """Insert W4A4 quantizers after Linear layers"""
    
    def _recursive_insert(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, nn.Sequential(
                    child,
                    ActivationQuantizer(num_bits=4)
                ))
            else:
                _recursive_insert(child)
    
    _recursive_insert(model)
    return model


def calibrate_activation_quantizers(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    num_batches: int = 10
):
    """Calibrate W4A4 quantizers"""
    
    print(f"  Calibrating W4A4 with {num_batches} batches...")
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break
            
            try:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'pixel_values': batch['pixel_values'].to(device)
                }
                
                if 'image_grid_thw' in batch:
                    grid = batch['image_grid_thw'].to(device)
                    if grid.dim() == 3:
                        grid = grid.reshape(-1, 3)
                    inputs['image_grid_thw'] = grid
                
                _ = model(**inputs)
                
            except Exception as e:
                continue
    
    print(f"  ✓ W4A4 calibration completed")


class QVLM_Quantizer:
    """Main Q-VLM quantizer"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def quantize(
        self,
        calibration_loader: DataLoader,
        enable_veo: bool = True,
        enable_w4a4: bool = False,
        veo_iterations: int = 30
    ):
        """Apply Q-VLM quantization"""
        
        print("\n" + "="*80)
        print("Q-VLM QUANTIZATION")
        print("="*80)
        
        # VEO
        if enable_veo:
            print("\n[STEP 1/2] Vision Encoder Optimization...")
            
            if hasattr(self.model, 'visual'):
                vision_encoder = self.model.visual
            elif hasattr(self.model, 'vision_model'):
                vision_encoder = self.model.vision_model
            else:
                print("  ⚠️ No vision encoder found, skipping VEO")
                vision_encoder = None
            
            if vision_encoder is not None:
                veo = VisionEncoderOptimizer(vision_encoder, self.device)
                vision_encoder = veo.optimize(calibration_loader, num_iterations=veo_iterations)
        
        # W4A4
        if enable_w4a4:
            print("\n[STEP 2/2] W4A4 Activation Quantization...")
            
            self.model = insert_activation_quantizers(self.model)
            calibrate_activation_quantizers(self.model, calibration_loader, self.device)
            
            print("  ✓ W4A4 enabled")
        
        print("\n" + "="*80)
        print("Q-VLM QUANTIZATION COMPLETED")
        print("="*80 + "\n")
        
        return self.model