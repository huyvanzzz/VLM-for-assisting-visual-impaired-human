# File: test_dataset.py
#!/usr/bin/env python3

import argparse
import sys
sys.path.append('.')
import torch
import unittest
from unittest.mock import MagicMock
from PIL import Image
from src.data.wad_dataset import WADDataset

# --- MOCK OBJECTS (Giả lập môi trường) ---
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.padding_side = "right"
        self.eos_token = "</s>"

    def __call__(self, text, return_tensors="pt", **kwargs):
        # Fake tokenize: Chuyển text thành list số ngẫu nhiên
        length = len(text.split()) + 5
        return {
            'input_ids': torch.randint(10, 1000, (1, length)),
            'attention_mask': torch.ones((1, length))
        }

class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()

    def __call__(self, text, images, return_tensors="pt", truncation=False, padding=False):
        # Fake Processor Output
        # Input images là list, output pixel_values thường là Tensor (Batch, N_Crops, 3, H, W)
        num_crops = 5
        # Giả lập 1 ảnh -> 5 mảnh (LLaVA logic)
        pixel_values = torch.randn(1, num_crops, 3, 384, 384) 
        
        text_out = self.tokenizer(text)
        return {
            'input_ids': text_out['input_ids'],
            'attention_mask': text_out['attention_mask'],
            'pixel_values': pixel_values,
            'image_sizes': torch.tensor([[1000, 1000]])
        }

# --- TEST CASE ---
class TestWADDataset(unittest.TestCase):
    def setUp(self):
        # Setup fake data
        self.mock_meta = [{'frame_path': 'f1', 'QA': {'A': 'Go'}, 'area_type': 'Road'}]
        self.mock_index = {'f1': {0: {'shard': 'x', 'tar_path': 'y'}}}
        self.mock_bbox = {'f1': {0: []}}
        
        self.dataset = WADDataset(
            self.mock_meta, self.mock_index, self.mock_bbox, 
            MockProcessor(), MockTokenizer()
        )
        # Override hàm load ảnh thật để trả về ảnh đen (tránh lỗi file IO)
        self.dataset._load_frames = lambda path, ids: [Image.new('RGB', (100,100))]

    def test_getitem_shapes(self):
        print("\n[1] Testing __getitem__ Output Shapes...")
        sample = self.dataset[0]
        
        input_ids = sample['input_ids']
        labels = sample['labels']
        pixel_values = sample['pixel_values']
        
        print(f" -> Input IDs Shape: {input_ids.shape}")
        print(f" -> Labels Shape: {labels.shape}")
        print(f" -> Pixel Values Shape: {pixel_values.shape}")
        
        self.assertEqual(input_ids.shape, labels.shape, "Input và Label phải cùng độ dài!")
        self.assertTrue(torch.is_tensor(pixel_values), "Pixel values phải là Tensor")
        print(" ✅ Shapes: OK")

    def test_label_masking(self):
        print("\n[2] Testing Label Masking (Prompt = -100)...")
        sample = self.dataset[0]
        labels = sample['labels']
        
        # Kiểm tra: Phần đầu của label phải là -100 (vì là prompt)
        # Phần cuối phải là > 0 (vì là câu trả lời 'Go')
        
        print(f" -> First 5 labels: {labels[:5].tolist()}")
        print(f" -> Last 5 labels:  {labels[-5:].tolist()}")
        
        self.assertTrue((labels[:5] == -100).all(), "Lỗi: Prompt không được mask thành -100!")
        self.assertNotEqual(labels[-1].item(), -100, "Lỗi: Câu trả lời bị mask mất tiêu!")
        print(" ✅ Label Masking: OK")

if __name__ == '__main__':
    unittest.main()