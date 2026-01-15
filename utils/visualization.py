"""
Visualization utilities for model analysis and interpretation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_comparison(results_dict, title='Model Comparison', save_path=None):
    """
    Plot comparison of multiple models.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with model names as keys and accuracy scores as values
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    models = list(results_dict.keys())
    scores = list(results_dict.values())
    
    colors = ['#3498db' if 'ELM' in m and 'QELM' not in m else '#e74c3c' for m in models]
    
    plt.bar(range(len(models)), scores, color=colors, alpha=0.7)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_stability(stability_results, save_path=None):
    """
    Plot stability analysis results (box plots).
    
    Parameters
    ----------
    stability_results : dict
        Dictionary with model names as keys and result dicts as values
    save_path : str, optional
        Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = list(stability_results.keys())
    means = [stability_results[m]['mean'] for m in models]
    stds = [stability_results[m]['std'] for m in models]
    
    # Plot 1: Mean accuracy with error bars
    colors = ['#3498db' if 'ELM' in m and 'QELM' not in m else '#e74c3c' for m in models]
    ax1.bar(range(len(models)), means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Mean Accuracy', fontsize=12)
    ax1.set_title('Stability Analysis: Mean ± Std', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Box plot of all runs
    all_scores = [stability_results[m]['scores'] for m in models]
    bp = ax2.boxplot(all_scores, labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Stability Analysis: Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_activation_comparison(activation_results, save_path=None):
    """
    Compare different activation functions.
    
    Parameters
    ----------
    activation_results : dict
        Nested dictionary: {model_type: {activation: score}}
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    model_types = list(activation_results.keys())
    activations = list(activation_results[model_types[0]].keys())
    
    x = np.arange(len(activations))
    width = 0.35
    
    for i, model_type in enumerate(model_types):
        scores = [activation_results[model_type][act] for act in activations]
        plt.bar(x + i * width, scores, width, label=model_type, alpha=0.7)
    
    plt.xlabel('Activation Function', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Activation Function Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x + width / 2, activations, rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_weight_distribution(model, save_path=None):
    """
    Visualize weight distributions of a trained model.
    
    Parameters
    ----------
    model : fitted ELM or QELM model
        Trained model with weights
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Input weights
    axes[0].hist(model.input_weights.flatten(), bins=50, alpha=0.7, color='#3498db')
    axes[0].set_xlabel('Weight Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Input Weights Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Biases
    axes[1].hist(model.biases.flatten(), bins=50, alpha=0.7, color='#e74c3c')
    axes[1].set_xlabel('Bias Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Biases Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Output weights
    axes[2].hist(model.output_weights.flatten(), bins=50, alpha=0.7, color='#2ecc71')
    axes[2].set_xlabel('Weight Value', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Output Weights Distribution', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
