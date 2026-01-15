"""
Experiment 2: Robustness Analysis
This experiment analyzes model stability across multiple random initializations.
Results correspond to Table 2 in the paper.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from datetime import datetime
from models import ELM_Sigmoid, QELM_Sigmoid, ELM_Tanh, QELM_Tanh
from utils.data_loader import load_dataset
from utils.metrics import stability_analysis
import pandas as pd


def run_robustness_experiment(datasets=['iris', 'wine'], 
                              n_hidden=100,
                              n_runs=30):
    """
    Analyze model robustness across multiple initializations.
    
    Parameters
    ----------
    datasets : list of str
        List of dataset names to evaluate
    n_hidden : int
        Number of hidden neurons
    n_runs : int
        Number of runs with different random seeds
        
    Returns
    -------
    results_df : pandas.DataFrame
        Robustness analysis results
    """
    # Define models to compare
    model_classes = {
        'ELM-Sigmoid': ELM_Sigmoid,
        'QELM-Sigmoid': QELM_Sigmoid,
        'ELM-Tanh': ELM_Tanh,
        'QELM-Tanh': QELM_Tanh,
    }
    
    results = []
    
    print("="*80)
    print("EXPERIMENT 2: ROBUSTNESS ANALYSIS")
    print("="*80)
    print(f"Hidden neurons: {n_hidden}")
    print(f"Number of runs: {n_runs}")
    print(f"Datasets: {', '.join(datasets)}")
    print("="*80)
    
    for dataset_name in datasets:
        print(f"\nAnalyzing {dataset_name} dataset...")
        
        # Load dataset
        X_train, X_test, y_train, y_test, info = load_dataset(dataset_name, random_state=42)
        
        print(f"  Dataset info: {info['n_train']} train, {info['n_test']} test")
        
        for model_name, model_class in model_classes.items():
            print(f"  Testing {model_name}...")
            
            # Run stability analysis
            stability_results = stability_analysis(
                model_class,
                X_train, X_test, y_train, y_test,
                n_runs=n_runs,
                n_hidden=n_hidden
            )
            
            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Mean_Accuracy': stability_results['mean'],
                'Std_Accuracy': stability_results['std'],
                'Min_Accuracy': stability_results['min'],
                'Max_Accuracy': stability_results['max'],
                'Range': stability_results['max'] - stability_results['min']
            })
            
            print(f"    Mean: {stability_results['mean']:.4f} "
                  f"± {stability_results['std']:.4f} "
                  f"(Range: {stability_results['max'] - stability_results['min']:.4f})")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY: Robustness Comparison")
    print("="*80)
    print("\nMean Accuracy (higher is better):")
    summary_mean = results_df.groupby('Model')['Mean_Accuracy'].mean().sort_values(ascending=False)
    print(summary_mean.to_string())
    
    print("\nStandard Deviation (lower is better - more stable):")
    summary_std = results_df.groupby('Model')['Std_Accuracy'].mean().sort_values()
    print(summary_std.to_string())
    
    # Calculate improvement
    print("\n" + "="*80)
    print("QELM IMPROVEMENT OVER ELM")
    print("="*80)
    for dataset in datasets:
        dataset_results = results_df[results_df['Dataset'] == dataset]
        elm_sig = dataset_results[dataset_results['Model'] == 'ELM-Sigmoid'].iloc[0]
        qelm_sig = dataset_results[dataset_results['Model'] == 'QELM-Sigmoid'].iloc[0]
        
        acc_improvement = (qelm_sig['Mean_Accuracy'] - elm_sig['Mean_Accuracy']) * 100
        std_improvement = (elm_sig['Std_Accuracy'] - qelm_sig['Std_Accuracy']) / elm_sig['Std_Accuracy'] * 100
        
        print(f"\n{dataset}:")
        print(f"  Accuracy improvement: {acc_improvement:+.2f}%")
        print(f"  Stability improvement: {std_improvement:+.2f}% (std reduction)")
    
    return results_df


def main():
    """Main function to run the experiment."""
    # Run experiment
    results_df = run_robustness_experiment(
        datasets=['iris', 'wine', 'breast_cancer'],
        n_hidden=100,
        n_runs=30
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'experiment_2_robustness_{timestamp}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    return results_df


if __name__ == '__main__':
    main()
