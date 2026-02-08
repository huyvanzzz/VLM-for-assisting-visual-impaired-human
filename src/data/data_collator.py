import torch
from typing import List, Dict

class VLMDataCollator:
    """Custom data collator for VLM models"""
    
    def __init__(self):
        self._debug_printed = False
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch with proper padding"""
        
        max_input_len = max([item['input_ids'].shape[0] for item in batch])
        max_label_len = max([item['labels'].shape[0] for item in batch])
        
        # Pad text
        input_ids = torch.stack([
            torch.nn.functional.pad(item['input_ids'], (0, max_input_len - item['input_ids'].shape[0]))
            for item in batch
        ])
        
        attention_mask = torch.stack([
            torch.nn.functional.pad(item['attention_mask'], (0, max_input_len - item['attention_mask'].shape[0]))
            for item in batch
        ])
        
        labels = torch.stack([
            torch.nn.functional.pad(
                item['labels'], 
                (0, max_label_len - item['labels'].shape[0]), 
                value=-100
            )
            for item in batch
        ])
        
        # Pixel values
        pixel_values_list = [item['pixel_values'] for item in batch]
        
        # Debug first batch
        if not self._debug_printed:
            for i, pv in enumerate(pixel_values_list):
                print(f"  Sample {i} pixel_values: {pv.shape}")
            self._debug_printed = True
        
        # Check shape consistency
        shapes = [pv.shape for pv in pixel_values_list]
        if len(set(shapes)) > 1:
            print(f"Inconsistent pixel_values shapes: {shapes}")
            max_tiles = max([pv.shape[0] for pv in pixel_values_list])
            pixel_values_list = [
                torch.nn.functional.pad(pv, (0, 0, 0, 0, 0, 0, 0, max_tiles - pv.shape[0]))
                if pv.shape[0] < max_tiles else pv[:max_tiles]
                for pv in pixel_values_list
            ]
        
        pixel_values = torch.cat(pixel_values_list, dim=0)
        
        # Build result
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': labels
        }
        
        # Handle image_sizes
        image_sizes = []
        for item in batch:
            if 'image_sizes' in item:
                sizes = item['image_sizes']
                
                if isinstance(sizes, torch.Tensor):
                    if sizes.dim() == 1:
                        image_sizes.append(tuple(sizes.tolist()))
                    else:
                        for size in sizes:
                            image_sizes.append(tuple(size.tolist()))
                elif isinstance(sizes, (tuple, list)):
                    if isinstance(sizes[0], (int, float)):
                        image_sizes.append(tuple(sizes))
                    else:
                        image_sizes.extend(sizes)
        
        if image_sizes:
            result['image_sizes'] = image_sizes
        
        # Handle image_grid_thw
        if 'image_grid_thw' in batch[0]:
            try:
                grid_list = []
                for item in batch:
                    grid = item['image_grid_thw']
                    
                    if grid.dim() == 1:
                        grid = grid.unsqueeze(0)
                    
                    grid_list.append(grid)
                
                result['image_grid_thw'] = torch.cat(grid_list, dim=0)
            
            except Exception as e:
                print(f"Warning: image_grid_thw collate failed: {e}")
        
        return result