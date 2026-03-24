import argparse
import yaml
import os
import json
import torch
import pickle
from collections import defaultdict
from torch.utils.data import DataLoader
from peft import PeftModel
from tqdm import tqdm
from datasets import load_dataset
from PIL import UnidentifiedImageError

import sys
sys.path.append('.')

# Import từ source code của bạn
from src.models.model_registry import build_model
from src.data.wad_dataset import WADDataset
from src.data.preprocessing import construct_prompt
from src.data.data_collator import VLMDataCollator

# ==========================================
# GHI ĐÈ DATASET ĐỂ CHỈ LẤY PROMPT (BỎ GROUND TRUTH)
# ==========================================
class WADInferenceDataset(WADDataset):
    def __getitem__(self, idx):
        try:
            sample = self.metadata[idx]
            folder_id = str(sample.get('folder_id', sample.get('frame_path')))
            target_frame_id = int(sample['frame_id'])
            
            # 1. Load ảnh và bbox (Dùng y hệt logic gốc của bạn)
            frame_ids = self._get_target_frames(folder_id, target_frame_id)
            frames = self._load_frames(folder_id, frame_ids)
            polm_list = self._load_bboxes(folder_id, frame_ids)
            
            # 2. Tạo Text Prompt (Nạp toàn bộ môi trường động vào prompt)
            messages = construct_prompt(polm_list, num_images=self.num_frames, metadata=sample) 
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 3. Xử lý Prompt + Image qua Processor (KHÔNG CÓ ANSWER TEXT VÀ LABELS)
            inputs = self.processor(
                text=prompt_text,
                images=frames,
                return_tensors="pt",
                truncation=False,
                padding=False
            )
            
            return_dict = {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'sample_idx': idx  # Truyền thêm idx để lúc lưu file biết kết quả của sample nào
            }
            
            if 'image_sizes' in inputs:
                return_dict['image_sizes'] = inputs['image_sizes'].squeeze(0)
            if 'image_grid_thw' in inputs:
                return_dict['image_grid_thw'] = inputs['image_grid_thw'].squeeze(0)
            
            return return_dict

        except (UnidentifiedImageError, OSError, IOError, Exception) as e:
            # Trong Inference, nếu lỗi thì trả về None để bỏ qua, không lấy random
            print(f"\n⚠️ Lỗi ở sample {idx} (folder_id: {folder_id}, frame_id: {target_frame_id}): {str(e)}")
            return None

# ==========================================
# CÁC HÀM XỬ LÝ CHÍNH
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Navigation VLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to LoRA checkpoint.")
    parser.add_argument("--input_data", type=str, required=False, help="File JSON chứa list data cần inference")
    parser.add_argument("--output_file", type=str, default="inference_results.json")
    return parser.parse_args()

def prepare_auxiliary_data(config):
    # 1. Load frame index
    print("Loading frame index...")
    with open("./wad_dataset/frame_index.pkl", 'rb') as f:
        frame_index = pickle.load(f)

    # 2. Load bboxes (CHÚ Ý: Lấy đúng file all_bboxes_2.jsonl của bạn)
    print("Loading bboxes from all_bboxes.jsonl...")
    bbox_file = "all_bboxes.jsonl"
    if os.path.exists(bbox_file):
        bbox_dataset = load_dataset("json", data_files=bbox_file, split="train")
    else:
        bbox_dataset = load_dataset(config['data']['name'], data_files=bbox_file, split="train")

    bbox_by_folder = defaultdict(lambda: defaultdict(list))
    for entry in bbox_dataset:
        folder_id = entry['folder_id']
        frame_id = entry['frame_id']
        bbox_by_folder[folder_id][frame_id].append({
            'label': entry['label'],
            'confidence': entry['probs'],
            'bbox': entry['boxs'],
            'relative_position': entry.get('relative_position', "unknown"),
            'distance_zone': entry.get('distance_zone', 'unknown'),
            'coming_to_user': entry.get('coming_to_user', False),
            'speed': entry.get('speed', 0.0),
            'danger_score': entry.get('danger_score', 0.0),
        })
    return frame_index, bbox_by_folder

def custom_collate_fn(batch, tokenizer):
    # Lọc bỏ các sample bị lỗi (None)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # Tách sample_idx ra khỏi batch
    sample_indices = [b.pop('sample_idx') for b in batch]
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    collated_batch = {
        'input_ids': torch.nn.utils.rnn.pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=pad_token_id),
        'attention_mask': torch.nn.utils.rnn.pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0),
    }
    
    # Xử lý pixel_values
    if 'pixel_values' in batch[0]:
        collated_batch['pixel_values'] = torch.cat([b['pixel_values'] for b in batch], dim=0)
    
    # Xử lý image_sizes (Thêm unsqueeze để giữ shape)
    if 'image_sizes' in batch[0]:
        sizes_list = []
        for b in batch:
            size = b['image_sizes']
            if isinstance(size, torch.Tensor) and size.dim() == 1:
                size = size.unsqueeze(0)
            sizes_list.append(size)
        collated_batch['image_sizes'] = torch.cat(sizes_list, dim=0)
        
    # Xử lý image_grid_thw (KHẮC PHỤC LỖI INDEX Ở ĐÂY)
    if 'image_grid_thw' in batch[0]:
        grid_list = []
        for b in batch:
            grid = b['image_grid_thw']
            if isinstance(grid, torch.Tensor) and grid.dim() == 1:
                grid = grid.unsqueeze(0)  # Kéo nó lại thành 2D (1, 3)
            grid_list.append(grid)
        collated_batch['image_grid_thw'] = torch.cat(grid_list, dim=0)
    
    # Nhét lại indices vào batch để sau này map với file output
    collated_batch['sample_indices'] = sample_indices
    return collated_batch

def extract_instruction(raw_text: str) -> str:
    text = raw_text.strip()
    if "<answer>" in text: text = text.split("<answer>")[-1]
    if "</answer>" in text: text = text.split("</answer>")[0]
    try:
        data = json.loads(text.strip())
        return str(data.get("instruction", "")).strip()
    except:
        return text.strip()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.checkpoint and 'model' in config and 'lora' in config['model']:
        config['model']['lora']['enabled'] = False
        
    print("Building Model...")
    vlm_wrapper = build_model(config)
    model, tokenizer, processor = vlm_wrapper.model, vlm_wrapper.tokenizer, vlm_wrapper.processor
    
    if args.checkpoint:
        print(f"Loading LoRA from {args.checkpoint}...")
        model = PeftModel.from_pretrained(
            model, args.checkpoint,
            torch_dtype=torch.bfloat16 if config.get('training', {}).get('bf16') else torch.float16,
            is_trainable=False
        )
    
    device = model.device
    model.eval()

    # --- ĐỌC DỮ LIỆU ---
    frame_index, bbox_by_folder = prepare_auxiliary_data(config)
    
    # Thay vì load file local, mình load trực tiếp từ repo Hugging Face của bạn
    print(f"Loading inference data (all_folder_frame.jsonl) from Hugging Face...")
    dataset_dict = load_dataset(
        config['data']['name'], 
        data_files={"test": "all_folder_frame.jsonl"} # Tên file trên HF của bạn
    )
    raw_metadata = dataset_dict["test"]
    
    image_size = None if config['model']['architecture'] == 'qwen' else tuple(config['model']['vision']['image_size'])
    
    # KHỞI TẠO DATASET KẾ THỪA
    target_dataset = WADInferenceDataset(
        metadata_dataset=dataset_dict,
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        processor=processor,
        tokenizer=tokenizer,
        split='test',
        num_frames=config['data'].get('num_frames', 1),
        image_size=image_size
    )
    
    eval_dataloader = DataLoader(
        target_dataset, 
        batch_size=1, # Giữ batch_size = 1 để inference an toàn
        shuffle=False, 
        collate_fn=lambda b: custom_collate_fn(b, tokenizer)
    )

    gen_config = {
        "max_new_tokens": 256,
        "do_sample": False,
        "num_beams": 3,
        "repetition_penalty": 1.3,
        "use_cache": True,
    }
    
    results = []
    print("\nBắt đầu chạy Inference...")
    print_count = 0
    for batch in tqdm(eval_dataloader, desc="Inferencing"):
        if batch is None:
            continue # Bỏ qua nếu data lỗi

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
        sample_indices = batch.pop('sample_indices')

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        if 'pixel_values' in batch:
            inputs['pixel_values'] = batch['pixel_values'].to(device)
            if 'image_grid_thw' in batch: inputs['image_grid_thw'] = batch['image_grid_thw'].to(device)
            if 'image_sizes' in batch: inputs['image_sizes'] = batch['image_sizes'].to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_config)
        
        prompt_length = input_ids.shape[1]
        
        # Xử lý kết quả cho từng sample trong batch (mặc dù hiện tại batch_size=1)
        for i, sample_idx in enumerate(sample_indices):
            generated_ids = outputs[i][prompt_length:]
            raw_output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            final_instruction = extract_instruction(raw_output_text)
            
            # Lấy lại metadata ban đầu để map kết quả
            original_sample = raw_metadata[sample_idx]
            folder_id = original_sample.get('folder_id', original_sample.get('frame_path', ""))
            frame_id = original_sample.get('frame_id', "")
            if print_count < 50:
                print(f"\n[{print_count+1}/50] Kết quả mẫu:")
                print(f"  - Thư mục : {folder_id}")
                print(f"  - Frame ID: {frame_id}")
                print(f"  - AI sinh : {final_instruction}")
                print("-" * 50)
                print_count += 1
            results.append({
                "folder_id": folder_id,
                "frame_id": frame_id,
                "instruction": final_instruction
            })

    # --- LƯU KẾT QUẢ ---
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\n✓ Xong! Đã lưu {len(results)} kết quả vào {args.output_file}")

if __name__ == "__main__":
    main()