import torch
from typing import List, Dict

class VLMDataCollator:
    """Custom data collator for VLM models"""
    
    # [FIX 1]: Phải truyền tokenizer vào để lấy đúng ID padding
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer 
        self._debug_printed = False
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch with proper padding"""
        
        # Lọc bỏ mẫu lỗi (None) nếu có
        batch = [x for x in batch if x is not None]
        if len(batch) == 0: return {}

        # --- PHẦN XỬ LÝ TEXT (Sửa để dùng đúng pad_token_id) ---
        input_ids_list = [item['input_ids'] for item in batch]
        labels_list = [item['labels'] for item in batch]
        attention_mask_list = [item['attention_mask'] for item in batch]
        
        # Dùng pad_sequence của PyTorch cho an toàn và chính xác
        # [FIX 2]: Thay vì pad=0, dùng self.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        
        # Attention mask pad bằng 0 là đúng (giữ nguyên logic, đổi cách viết cho gọn)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, 
            batch_first=True, 
            padding_value=0
        )
        
        # Labels pad bằng -100 là đúng
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, 
            batch_first=True, 
            padding_value=-100
        )
        
        # --- PHẦN XỬ LÝ ẢNH (Pixel Values) ---
        # [FIX 3]: Bỏ đoạn logic "Check shape consistency" và padding pixel thủ công.
        # Lý do: Qwen2-VL nối đuôi các patch ảnh lại (flatten), không cần các ảnh phải cùng kích thước tensor.
        # Việc padding pixel sẽ làm lệch 'image_grid_thw'.
        
        pixel_values_list = [item['pixel_values'] for item in batch]
        pixel_values = torch.cat(pixel_values_list, dim=0) # Chỉ cần cat lại là xong
        
        # Build result
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': labels
        }
        
        # --- PHẦN CÒN LẠI (Giữ nguyên của bạn) ---
        
        # Handle image_sizes (Giữ nguyên logic của bạn)
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
        
        # Handle image_grid_thw (Giữ nguyên logic của bạn - cái này viết đúng rồi)
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