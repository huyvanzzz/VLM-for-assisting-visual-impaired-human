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
        image_size: tuple = (336, 336)
    ):
        self.metadata = metadata_dataset[split]
        self.frame_index = frame_index
        self.bbox_by_folder = bbox_by_folder
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        
    def __len__(self):
        return len(self.metadata)
    
    def _load_frames(self, frame_path: str, frame_ids: List[int]) -> List[Image.Image]:
        """Load multiple frames with fixed size"""
        
        # Group frames by shard
        shard_to_frames = {}
        for frame_id in frame_ids:
            if frame_id not in self.frame_index[frame_path]:
                raise ValueError(f"Frame {frame_id} not in index")
            
            frame_info = self.frame_index[frame_path][frame_id]
            shard_path = frame_info['shard']
            
            if shard_path not in shard_to_frames:
                shard_to_frames[shard_path] = []
            shard_to_frames[shard_path].append((frame_id, frame_info['tar_path']))
        
        # Load frames
        frames_dict = {}
        for shard_path, frame_list in shard_to_frames.items():
            with tarfile.open(shard_path, 'r') as tar:
                for frame_id, tar_path in frame_list:
                    member = tar.getmember(tar_path)
                    file_obj = tar.extractfile(member)
                    img_bytes = file_obj.read()
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    
                    frames_dict[frame_id] = img
        
        return [frames_dict[fid] for fid in frame_ids]
    
    def _load_bboxes(self, frame_path: str, frame_ids: List[int]) -> List[POLMData]:
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
        """Safe frame selection with fallback"""
        
        if frame_path not in self.frame_index:
            raise ValueError(f"Frame path not in index: {frame_path}")
        
        available_frames = sorted(self.frame_index[frame_path].keys())
        
        if len(available_frames) == 0:
            raise ValueError(f"No frames in {frame_path}")
        
        # Single frame: take last
        if num_frames == 1:
            return [available_frames[-1]]
        
        # Multiple frames: even spacing
        if len(available_frames) >= num_frames:
            indices = np.linspace(0, len(available_frames) - 1, num_frames, dtype=int)
            return [available_frames[i] for i in indices]
        
        # Pad with last frame
        selected = available_frames.copy()
        while len(selected) < num_frames:
            selected.append(available_frames[-1])
        
        return selected[:num_frames]
        
    def __getitem__(self, idx):
        sample = self.metadata[idx]
        frame_path = sample['frame_path']
        
        # Select frames
        frame_ids = self._select_frames_safe(frame_path, num_frames=self.num_frames)
        
        # Load frames and bboxes
        frames = self._load_frames(frame_path, frame_ids)
        polm_list = self._load_bboxes(frame_path, frame_ids)
        
        # Construct prompt
        prompt = construct_prompt(polm_list, num_images=self.num_frames)
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=frames,
            return_tensors="pt",
            padding=True
        )
        
        # Ground truth
        ground_truth_dict = map_metadata_to_ground_truth(sample)
        ground_truth_json = ground_truth_dict.to_json()
        
        # Tokenize ground truth
        gt_tokens = self.tokenizer(
            ground_truth_json,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=128,
            add_special_tokens=False
        )['input_ids'].squeeze(0)
        
        # Create labels
        input_ids = inputs['input_ids'].squeeze(0)
        prompt_len = len(input_ids)
        gt_len = len(gt_tokens)
        
        labels = torch.full((prompt_len,), -100, dtype=torch.long)
        
        if gt_len <= prompt_len:
            labels[-gt_len:] = gt_tokens
        else:
            labels[:] = gt_tokens[:prompt_len]
        
        # Build return dict
        return_dict = {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': labels
        }
        
        # Handle optional keys
        if 'image_sizes' in inputs:
            image_sizes = inputs['image_sizes'].squeeze(0)
            return_dict['image_sizes'] = image_sizes
        
        if 'image_grid_thw' in inputs:
            grid = inputs['image_grid_thw'].squeeze(0)
            if grid.dim() == 1:
                grid = grid.unsqueeze(0)
            return_dict['image_grid_thw'] = grid
        
        if 'batch_num_images' in inputs:
            return_dict['batch_num_images'] = inputs['batch_num_images'].squeeze(0)
        
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