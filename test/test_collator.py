# File: test_collator.py
#!/usr/bin/env python3

import argparse
import sys
sys.path.append('.')
import torch
import unittest
from src.data.data_collator import VLMDataCollator

class TestCollator(unittest.TestCase):
    
    def test_batching(self):
        print("\n[1] Testing Batch Collation...")
        collator = VLMDataCollator()
        
        # Tạo 2 sample giả: 1 ngắn, 1 dài
        # Sample 1: Text dài 10, Ảnh 5 mảnh
        sample1 = {
            'input_ids': torch.randint(0, 100, (10,)),
            'labels': torch.full((10,), -100),
            'pixel_values': torch.randn(5, 3, 384, 384), # 5 mảnh
            'attention_mask': torch.ones(10),
            'image_sizes': torch.tensor([100, 100])
        }
        
        # Sample 2: Text dài 20, Ảnh 5 mảnh
        sample2 = {
            'input_ids': torch.randint(0, 100, (20,)),
            'labels': torch.full((20,), -100),
            'pixel_values': torch.randn(5, 3, 384, 384),
            'attention_mask': torch.ones(20),
            'image_sizes': torch.tensor([100, 100])
        }
        
        batch = collator([sample1, sample2])
        
        print(f" -> Batch Keys: {list(batch.keys())}")
        
        # 1. Check Padding Text
        # Batch input_ids phải có size (2, 20) (theo thằng dài nhất)
        self.assertEqual(batch['input_ids'].shape, (2, 20))
        print(f" -> Input IDs Padded: {batch['input_ids'].shape} (Expected (2, 20))")
        
        # 2. Check Pixel Values Flattening
        # Tổng số mảnh ảnh = 5 (sample1) + 5 (sample2) = 10
        # Collator của LLaVA thường nối liền (concatenate) dim 0
        expected_pixels = 10
        self.assertEqual(batch['pixel_values'].shape[0], expected_pixels)
        print(f" -> Pixel Values Stacked: {batch['pixel_values'].shape} (Expected (10, 3, 384, 384))")
        
        print(" ✅ Batch Logic: OK")

if __name__ == '__main__':
    unittest.main()