import argparse
import yaml
import os
import json
import torch
import pickle
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from peft import PeftModel
from tqdm import tqdm
import sys
sys.path.append('.')

# Import project modules
from src.models.model_registry import build_model
from src.data.wad_dataset import build_dataset, WADDataset
from src.data.data_collator import VLMDataCollator
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference and Extract Instructions")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint folder.")
    parser.add_argument("--output_file", type=str, default="inference_results.json", help="Path to save results")
    parser.add_argument("--split", type=str, default="test_alter", choices=["train", "valid", "test_alter", "test_QA"])
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick testing")
    return parser.parse_args()

def prepare_auxiliary_data(config):
    index_file = "./wad_dataset/frame_index.pkl"
    with open(index_file, 'rb') as f:
        frame_index = pickle.load(f)

    bbox_file = "all_bboxes.jsonl"
    if os.path.exists(bbox_file):
        bbox_dataset = load_dataset("json", data_files=bbox_file, split="train")
    else:
        bbox_dataset = load_dataset(config['data']['name'], data_files="all_bboxes.jsonl", split="train")

    bbox_by_folder = defaultdict(lambda: defaultdict(list))
    for bbox_entry in bbox_dataset:
        folder_id = bbox_entry['folder_id']
        frame_id = bbox_entry['frame_id']
        
        bbox_by_folder[folder_id][frame_id].append({
            'label': bbox_entry['label'],
            'confidence': bbox_entry['probs'],
            'bbox': bbox_entry['boxs'],
            'relative_position': bbox_entry.get('relative_position', "unknown"),
            'distance_zone': bbox_entry.get('distance_zone', 'unknown'),
            'coming_to_user': bbox_entry.get('coming_to_user', False),
            'speed': bbox_entry.get('speed', 0.0),
            'danger_score': bbox_entry.get('danger_score', 0.0),
        })
    return frame_index, bbox_by_folder

def extract_instruction(raw_text: str) -> str:
    """Hàm bóc tách 'instruction' từ text do model sinh ra."""
    text = raw_text.strip()
    
    if "<answer>" in text:
        text = text.split("<answer>")[-1]
    if "</answer>" in text:
        text = text.split("</answer>")[0]
    text = text.strip()
    
    try:
        data = json.loads(text)
        return str(data.get("instruction", "")).strip()
    except json.JSONDecodeError:
        return text

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
            torch_dtype=torch.bfloat16 if config['training'].get('bf16', False) else torch.float16,
            is_trainable=False
        )
    
    device = config.get('hardware', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print(f"Loading dataset: {args.split}...")
    raw_data_list = None
    
    if args.split in ["train", "valid"]:
        train_dataset, valid_dataset = build_dataset(config, processor, tokenizer)
        target_dataset = train_dataset if args.split == "train" else valid_dataset
    else:
        frame_index, bbox_by_folder = prepare_auxiliary_data(config)
        data_file = "test_alter.json" if args.split == "test_alter" else "test_QA.json"
        
        dataset_dict = load_dataset(config['data']['name'], data_files={"test": data_file})
        raw_data_list = [item for item in dataset_dict["test"]]
        
        image_size = None if config['model']['architecture'] == 'qwen' else tuple(config['model']['vision']['image_size'])
        target_dataset = WADDataset(
            metadata_dataset=dataset_dict, frame_index=frame_index, bbox_by_folder=bbox_by_folder,
            processor=processor, tokenizer=tokenizer, split='test',
            num_frames=config['data'].get('num_frames', 1), image_size=image_size
        )
    
    if args.max_samples:
        target_dataset = Subset(target_dataset, range(args.max_samples))
        if raw_data_list: raw_data_list = raw_data_list[:args.max_samples]

    data_collator = VLMDataCollator(tokenizer=tokenizer)
    eval_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

    gen_config = {
        "max_new_tokens": 256,
        "num_beams": 3,
        "do_sample": False,
        "repetition_penalty": 1.3,
        "use_cache": True
    }
    
    results = []
    print("\nStarting Inference...")
    
    for i, batch in enumerate(tqdm(eval_dataloader, desc="Inferencing")):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        valid_label_indices = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if len(valid_label_indices) == 0:
            prompt_ids = input_ids[0]
        else:
            prompt_ids = input_ids[0][:valid_label_indices[0].item()]

        single_input = {
            'input_ids': prompt_ids.unsqueeze(0),
            'attention_mask': torch.ones_like(prompt_ids.unsqueeze(0))
        }
        
        if 'pixel_values' in batch:
            single_input['pixel_values'] = batch['pixel_values'].to(device)
            if 'image_grid_thw' in batch: single_input['image_grid_thw'] = batch['image_grid_thw'].to(device)
            if 'image_sizes' in batch: single_input['image_sizes'] = batch['image_sizes'].to(device)

        with torch.no_grad():
            outputs = model.generate(**single_input, **gen_config)
        
        generated_ids = outputs[0][len(prompt_ids):]
        raw_output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        final_instruction = extract_instruction(raw_output_text)
        
        # --- ĐÃ SỬA LẠI FORMAT Ở ĐÂY ---
        result_item = {}
        if raw_data_list is not None and i < len(raw_data_list):
            item_data = raw_data_list[i]
            
            # Gắn trực tiếp 2 trường riêng biệt
            if 'folder_id' in item_data:
                result_item["folder_id"] = item_data['folder_id']
            if 'frame_id' in item_data:
                result_item["frame_id"] = item_data['frame_id']
                
            # Đề phòng trường hợp file json chỉ có key 'id'
            if 'folder_id' not in item_data and 'frame_id' not in item_data and 'id' in item_data:
                result_item["id"] = str(item_data['id'])

        # Fallback nếu không có data nào khớp
        if not result_item:
            result_item["id"] = str(i)

        # Gắn instruction vào cuối
        result_item["instruction"] = final_instruction
        results.append(result_item)

    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\n✓ DONE! Saved {len(results)} items to {args.output_file}")

if __name__ == "__main__":
    main()