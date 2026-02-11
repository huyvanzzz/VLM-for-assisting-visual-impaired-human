# File: test_preprocessing.py
#!/usr/bin/env python3

import argparse
import sys
sys.path.append('.')
import unittest
from src.data.preprocessing import map_metadata_to_ground_truth, construct_prompt, POLMData

class TestPreprocessing(unittest.TestCase):
    
    def test_1_polm_data(self):
        print("\n[1] Testing POLM Data Structure...")
        # Giả lập 1 object detected
        obj = POLMData(object_type="car", bbox=[0, 0, 50, 50], confidence=0.95)
        text = obj.to_text()
        print(f" -> Output: {text}")
        
        self.assertIn("Object: car", text)
        self.assertIn("Confidence: 0.95", text)
        print(" ✅ POLM Data: OK")

    def test_2_metadata_mapping(self):
        print("\n[2] Testing Metadata Mapping...")
        # Giả lập dữ liệu thô từ file JSON
        raw_metadata = {
            'area_type': 'Busy Street',
            'weather_condition': 'Sunny',
            'traffic_flow_rating': 'High',
            'summary': 'A crowded street',
            'QA': {'A': 'Walk slowly'}
        }
        
        gt = map_metadata_to_ground_truth(raw_metadata)
        print(f" -> Input Area: 'Busy Street' | Output Location: '{gt.location}'")
        print(f" -> Instruction: '{gt.instruction}'")
        
        self.assertEqual(gt.location, 'busy_street') # Check map đúng key không
        self.assertEqual(gt.traffic, 'high')
        self.assertEqual(gt.instruction, 'Walk slowly')
        print(" ✅ Metadata Mapping: OK")

    def test_3_construct_prompt(self):
        print("\n[3] Testing Prompt Construction...")
        objs = [POLMData("person", [0,0,10,10], 0.9)]
        prompt = construct_prompt(objs, num_images=1)
        
        print(f" -> Prompt Preview:\n{'-'*20}\n{prompt}\n{'-'*20}")
        
        self.assertIn("<image>", prompt) # Phải có token ảnh
        self.assertIn("Perception:", prompt) # Phải có Chain-of-Thought
        self.assertIn("Respond in JSON", prompt)
        print(" ✅ Prompt format: OK")

if __name__ == '__main__':
    unittest.main()