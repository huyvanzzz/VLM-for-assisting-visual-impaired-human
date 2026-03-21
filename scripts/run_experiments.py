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
import sys
sys.path.append('.')

# Import project modules
from src.models.model_registry import build_model
from src.data.wad_dataset import WADDataset
from src.data.data_collator import VLMDataCollator
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Pure Inference for Navigation VLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to LoRA checkpoint.")
    # Truyền thẳng file data thực tế vào đây, không test_alter gì nữa
    parser.add_argument("--input_data", type=str, required=True, help="Path to your real data JSON file") 
    parser.add_argument("--output_file", type=str, default="inference_results.json")
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
    text = raw_text.strip()
    if "<answer>" in text: text = text.split("<answer>")[-1]
    if "</answer>" in text: text = text.split("</answer>")[0]
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

    # --- ĐỌC DỮ LIỆU THỰC TẾ ---
    print(f"Loading real data from: {args.input_data}...")
    frame_index, bbox_by_folder = prepare_auxiliary_data(config)
    
    dataset_dict = load_dataset("json", data_files={"test": args.input_data})
    raw_data_list = [item for item in dataset_dict["test"]]
    
    image_size = None if config['model']['architecture'] == 'qwen' else tuple(config['model']['vision']['image_size'])
    target_dataset = WADDataset(
        metadata_dataset=dataset_dict, frame_index=frame_index, bbox_by_folder=bbox_by_folder,
        processor=processor, tokenizer=tokenizer, split='test',
        num_frames=config['data'].get('num_frames', 1), image_size=image_size
    )
    
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
        # TRONG INFERENCE THỰC TẾ: Không có labels, input_ids chính là toàn bộ prompt đầu vào.
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

        single_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if 'pixel_values' in batch:
            single_input['pixel_values'] = batch['pixel_values'].to(device)
            if 'image_grid_thw' in batch: single_input['image_grid_thw'] = batch['image_grid_thw'].to(device)
            if 'image_sizes' in batch: single_input['image_sizes'] = batch['image_sizes'].to(device)

        with torch.no_grad():
            outputs = model.generate(**single_input, **gen_config)
        
        # Bỏ qua phần prompt ban đầu, chỉ lấy những token mới được model sinh ra
        prompt_length = input_ids.shape[1]
        generated_ids = outputs[0][prompt_length:]
        raw_output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        final_instruction = extract_instruction(raw_output_text)
        
        # Gắn ID
        result_item = {}
        if i < len(raw_data_list):
            item_data = raw_data_list[i]
            if 'folder_id' in item_data: result_item["folder_id"] = item_data['folder_id']
            if 'frame_id' in item_data: result_item["frame_id"] = item_data['frame_id']
                
        if not result_item:
            result_item["id"] = str(i)

        result_item["instruction"] = final_instruction
        results.append(result_item)

    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\n✓ DONE! Saved {len(results)} items to {args.output_file}")

if __name__ == "__main__":
    main()