import json
from typing import Dict, List
from collections import defaultdict
import numpy as np

def exact_match_score(pred: str, gold: str) -> float:
    """Exact match (0 or 1)"""
    return 1.0 if pred.strip() == gold.strip() else 0.0

def field_accuracy(pred_json: Dict, gold_json: Dict, field: str) -> float:
    """Accuracy for a specific field"""
    pred_val = pred_json.get(field, "")
    gold_val = gold_json.get(field, "")
    return 1.0 if pred_val == gold_val else 0.0

def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute all metrics"""
    
    scores = defaultdict(list)
    
    for pred, ref in zip(predictions, references):
        try:
            # Parse JSON
            pred_json = json.loads(pred) if isinstance(pred, str) else pred
            ref_json = json.loads(ref) if isinstance(ref, str) else ref
            
            # Exact match
            scores['exact_match'].append(exact_match_score(pred, ref))
            
            # Field accuracies
            for field in ['location', 'weather', 'traffic', 'scene', 'instruction']:
                scores[f'{field}_accuracy'].append(field_accuracy(pred_json, ref_json, field))
        
        except json.JSONDecodeError:
            # Invalid JSON counts as 0
            scores['exact_match'].append(0.0)
            for field in ['location', 'weather', 'traffic', 'scene', 'instruction']:
                scores[f'{field}_accuracy'].append(0.0)
    
    # Average scores
    results = {key: np.mean(values) * 100 for key, values in scores.items()}
    
    return results

def evaluate_predictions(pred_file: str, ref_file: str) -> Dict[str, float]:
    """Evaluate predictions from files"""
    
    with open(pred_file, 'r') as f:
        predictions = [line.strip() for line in f]
    
    with open(ref_file, 'r') as f:
        references = [line.strip() for line in f]
    
    return compute_metrics(predictions, references)