import tarfile
from torch.utils.data import Dataset
from PIL import Image
import io
from typing import List, Dict
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from .preprocessing import POLMData, construct_prompt, map_metadata_to_ground_truth


class WADDataset(Dataset):
    def __init__(
        self,
        metadata_dataset,
        frame_index: dict,
        bbox_by_folder: dict,
        processor,
        tokenizer,
        split: str = 'train',
        num_frames: int = 1,
        image_size: tuple = (384, 384)
    ):
        self.metadata = metadata_dataset[split]
        self.frame_index = frame_index
        self.bbox_by_folder = bbox_by_folder
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        
        # Cấu hình Tokenizer để tiết kiệm token
        self.tokenizer.padding_side = "right" # Quan trọng cho training
        self.tokenizer.truncation_side = "right" # Quan trọng cho training
    def __len__(self):
        return len(self.metadata)

    def _load_frames(self, frame_path: str, frame_ids: List[int]) -> List[Image.Image]:
        """Load và xử lý ảnh (Padding luôn tại đây)"""
        # (Giữ nguyên logic load tarfile của bạn để code gọn, chỉ thêm đoạn xử lý ảnh)
        shard_to_frames = {}
        for frame_id in frame_ids:
            if frame_id not in self.frame_index[frame_path]:
                raise ValueError(f"Frame {frame_id} not in index")
            frame_info = self.frame_index[frame_path][frame_id]
            shard_path = frame_info['shard']
            if shard_path not in shard_to_frames:
                shard_to_frames[shard_path] = []
            shard_to_frames[shard_path].append((frame_id, frame_info['tar_path']))
        
        frames_dict = {}
        for shard_path, frame_list in shard_to_frames.items():
            with tarfile.open(shard_path, 'r') as tar:
                for frame_id, tar_path in frame_list:
                    member = tar.getmember(tar_path)
                    file_obj = tar.extractfile(member)
                    img = Image.open(io.BytesIO(file_obj.read())).convert('RGB')
                    # -------------------------------------------------
                    frames_dict[frame_id] = img
        
        return [frames_dict[fid] for fid in frame_ids]

    def _load_bboxes(self, frame_path: str, frame_ids: List[int]) -> List[POLMData]:
        # (Giữ nguyên logic cũ của bạn)
        polm_list = []
        if frame_path not in self.bbox_by_folder:
            return polm_list
        for frame_id in frame_ids:
            if frame_id in self.bbox_by_folder[frame_path]:
                bboxes = self.bbox_by_folder[frame_path][frame_id]
                for bbox in bboxes:
                    polm = POLMData(
                        object_type=bbox['label'],
                        bbox=bbox['bbox'],
                        confidence=bbox['confidence']
                    )
                    polm_list.append(polm)
        return polm_list

    def _select_frames_safe(self, frame_path: str, num_frames: int = 1) -> List[int]:
        # (Giữ nguyên logic cũ của bạn)
        if frame_path not in self.frame_index:
            raise ValueError(f"Frame path not in index: {frame_path}")
        available_frames = sorted(self.frame_index[frame_path].keys())
        if len(available_frames) == 0:
            raise ValueError(f"No frames in {frame_path}")
        if num_frames == 1:
            return [available_frames[-1]]
        if len(available_frames) >= num_frames:
            indices = np.linspace(0, len(available_frames) - 1, num_frames, dtype=int)
            return [available_frames[i] for i in indices]
        selected = available_frames.copy()
        while len(selected) < num_frames:
            selected.append(available_frames[-1])
        return selected[:num_frames]

    def __getitem__(self, idx):
        sample = self.metadata[idx]
        frame_path = sample['frame_path']
        
        # 1. Load Data
        frame_ids = self._select_frames_safe(frame_path, num_frames=self.num_frames)
        frames = self._load_frames(frame_path, frame_ids)
        polm_list = self._load_bboxes(frame_path, frame_ids)
        
        # 2. Tạo Text
        prompt_text = construct_prompt(polm_list, num_images=self.num_frames)
        ground_truth_dict = map_metadata_to_ground_truth(sample)
        
        # Tối ưu Token: Chỉ lấy JSON string gọn nhất + Token kết thúc
        answer_text = ground_truth_dict.to_json() + self.tokenizer.eos_token

        # 3. Xử lý Prompt + Image qua Processor
        # Lưu ý: padding=False để tự xử lý ghép chuỗi thủ công cho chính xác
        inputs = self.processor(
            text=prompt_text,
            images=frames,
            return_tensors="pt",
            truncation=False,  # <--- THÊM DÒNG NÀY (Bắt buộc): Cấm processor tự cắt
            padding=False      # <--- THÊM DÒNG NÀY: Để mình tự xử lý padding sau
        )
        
        debug_pixel_values = inputs['pixel_values']
        
        # # In ra màn hình console (GIỮ NGUYÊN CỦA BẠN)
        # print(f"\n[DEBUG IMAGE INFO]")
        # print(f" - Shape gốc: {debug_pixel_values.shape}")
        # # Shape thường là: (Batch, Num_Crops, Channels, Height, Width)
        # # Ví dụ: torch.Size([1, 3, 3, 384, 384])
        
        # if len(debug_pixel_values.shape) == 5:
        #     n_crops = debug_pixel_values.shape[1]
        #     h = debug_pixel_values.shape[3]
        #     w = debug_pixel_values.shape[4]
        #     print(f" - Số lượng mảnh (Crops): {n_crops}")
        #     print(f" - Kích thước mỗi mảnh: {h} x {w}")
        # else:
        #     print(f" - Shape lạ: {debug_pixel_values.shape}")
            
        # Lấy các Tensor ra khỏi batch dimension (vì processor trả về batch=1)
        prompt_input_ids = inputs['input_ids'].squeeze(0)
        prompt_attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)

        # ==========================================================================
        # [FIX LỖI] CHÈN CODE SỬA LỖI TẠI ĐÂY (Trước khi ghép chuỗi)
        # ==========================================================================
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        num_text_tokens = (prompt_input_ids == image_token_id).sum().item()
        
        # 2. Đếm số người (Số mảnh ảnh thực tế model nhìn thấy)
        # pixel_values shape thường là: (Số_lượng_mảnh, Channels, H, W)
        num_vision_patches = pixel_values.shape[0] 
        
        # 3. Tính độ lệch
        diff = num_vision_patches - num_text_tokens
        
        # 4. In ra bằng chứng (Chỉ in khi có lệch để đỡ rác màn hình)
        if diff != 0:
            print(f"\n==========================================")
            print(f"🕵️ [CHECKER] PHÁT HIỆN LỆCH TOKEN TẠI INDEX {idx}")
            print(f"   - Text Prompt có:   {num_text_tokens} thẻ <image>")
            print(f"   - Pixel Values có:  {num_vision_patches} mảnh ảnh (patches)")
            print(f"   - Độ lệch (Diff):   {diff}")
            
            if diff == 1:
                print(f"   -> KẾT LUẬN: Thừa đúng 1 mảnh. Đây chính là GLOBAL VIEW (hoặc CLS context).")
                print(f"   -> Kích thước mảnh thừa (cái cuối cùng): {pixel_values[-1].shape}")
            else:
                print(f"   -> KẾT LUẬN: Lệch {diff} (Có thể do nhiều frame hoặc lỗi logic khác).")
            print(f"==========================================\n")
        
        print(f"\n[DEBUG ALIGNMENT CHECK - AUTO-FIX]")
        print("pixel_values shape:", pixel_values.shape)
        print("pixel_values:", pixel_values.shape[0])
        # Nếu là 3 mảnh (Features=2052) mà Token chỉ có 2051 -> Bù 1 token
        if len(pixel_values.shape) == 4:
            # print(" -> [AUTO-FIX] Phát hiện 2051 tokens (thiếu 1). Đang bù thêm 1 token <image>...")
            extra_token = torch.tensor([image_token_id], dtype=torch.long)
            extra_mask = torch.tensor([1], dtype=torch.long)
            
            # Nối vào đuôi Prompt
            prompt_input_ids = torch.cat([prompt_input_ids, extra_token], dim=0)
            prompt_attention_mask = torch.cat([prompt_attention_mask, extra_mask], dim=0)
        # ==========================================================================

        # 4. Tokenize Answer (Câu trả lời)
        answer_tokens = self.tokenizer(
            answer_text,
            return_tensors="pt",
            add_special_tokens=False, # Không thêm BOS nữa vì đã có ở Prompt rồi
            truncation=True,
            max_length=256 # Giới hạn độ dài câu trả lời để tiết kiệm bộ nhớ
        )
        answer_input_ids = answer_tokens['input_ids'].squeeze(0)
        
        # 5. GHÉP CHUỖI (CONCATENATE) -> Logic Training Chuẩn
        input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=0)
        
        # Tạo Attention Mask (1 cho cả prompt và answer)
        attention_mask = torch.cat([
            prompt_attention_mask,
            torch.ones_like(answer_input_ids)
        ], dim=0)
        
        # Tạo Labels
        # - Vùng Prompt: -100 (Model không cần học lại câu hỏi)
        # - Vùng Answer: ID của token (Model phải đoán câu trả lời)
        labels = torch.cat([
            torch.full((len(prompt_input_ids),), -100, dtype=torch.long),
            answer_input_ids
        ], dim=0)

        # 7. Đóng gói kết quả
        return_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': labels
        }
        
        # (GIỮ NGUYÊN CODE IN CỦA BẠN ĐỂ CHECK LẠI)
        image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

        num_image_tokens = (input_ids == image_token_id).sum().item()

        debug_count = getattr(self, '_debug_count', 0)  # Lấy giá trị hoặc 0

        # if debug_count < 2:
        #     print("\n[DEBUG ALIGNMENT CHECK]")
        #     print("Total input_ids length:", len(input_ids))
        #     print("Image token id:", image_token_id)
        #     print("Number of <image> tokens in text:", num_image_tokens) 
        #     print("Pixel_values shape:", pixel_values.shape)
            
        #     self._debug_count = debug_count + 1  # Lưu lại

        # Copy các thông tin phụ (quan trọng cho model Qwen/LLaVA)
        if 'image_sizes' in inputs:
            return_dict['image_sizes'] = inputs['image_sizes'].squeeze(0)
        if 'image_grid_thw' in inputs:
            return_dict['image_grid_thw'] = inputs['image_grid_thw'].squeeze(0)
            
        return return_dict


def build_dataset(config: Dict, processor, tokenizer):
    """Build train/eval datasets from config"""
    
    from datasets import load_dataset
    from collections import defaultdict
    import pickle
    import os
    
    # Load metadata
    print("Loading metadata...")
    metadata = load_dataset(
        config['data']['name'],
        data_files={
            "train": "train.json",
            "test": "test_alter.json"
        }
    )
    
    # Load bboxes
    print("Loading bboxes...")
    bbox_dataset = load_dataset(
        config['data']['name'],
        data_files="all_bboxes.jsonl",
        split="train"
    )
    
    bbox_by_folder = defaultdict(lambda: defaultdict(list))
    for bbox_entry in bbox_dataset:
        folder_id = bbox_entry['folder_id']
        frame_id = bbox_entry['frame_id']
        
        bbox_by_folder[folder_id][frame_id].append({
            'label': bbox_entry['label'],
            'confidence': bbox_entry['probs'],
            'bbox': bbox_entry['boxs']
        })
    
    # Load frame index
    print("Loading frame index...")
    index_file = "./wad_dataset/frame_index.pkl"
    
    if os.path.exists(index_file):
        with open(index_file, 'rb') as f:
            frame_index = pickle.load(f)
    else:
        raise FileNotFoundError(f"Frame index not found at {index_file}. Run build_frame_index.py first.")
    
    # Create datasets
    train_dataset = WADDataset(
        metadata_dataset=metadata,
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        processor=processor,
        tokenizer=tokenizer,
        split='train',
        num_frames=config['data']['num_frames'],
        image_size=tuple(config['model']['vision']['image_size'])
    )
    
    # Train/val split
    train_size = config['data']['train_split']
    indices = list(range(len(train_dataset)))
    
    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_size,
        random_state=config['data']['seed']
    )
    
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    print(f"✓ Train: {len(train_subset)}, Val: {len(val_subset)}")
    
    return train_subset, val_subset