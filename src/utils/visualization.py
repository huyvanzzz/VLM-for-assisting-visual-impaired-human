import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List

def plot_training_curves(log_history: List[Dict], output_path: str = None):
    """Plot training loss and eval metrics"""
    
    df = pd.DataFrame(log_history)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    if 'loss' in df.columns:
        axes[0].plot(df['step'], df['loss'], label='Train Loss')
    if 'eval_loss' in df.columns:
        axes[0].plot(df['step'], df['eval_loss'], label='Eval Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Evaluation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Learning rate
    if 'learning_rate' in df.columns:
        axes[1].plot(df['step'], df['learning_rate'])
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {output_path}")
    else:
        plt.show()

def plot_model_comparison(results: Dict[str, Dict[str, float]], output_path: str = None):
    """Plot comparison between different models"""
    
    df = pd.DataFrame(results).T
    
    df.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('Model')
    plt.ylabel('Score (%)')
    plt.title('Model Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {output_path}")
    else:
        plt.show()