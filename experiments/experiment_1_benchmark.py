"""
Experiment 1: Benchmark Comparison
This experiment compares traditional ELM and QELM variants across multiple datasets.
Results correspond to Table 1 in the paper.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from datetime import datetime
from models import (
    ELM_Sigmoid, ELM_Tanh, ELM_ReLU, ELM_Sine, ELM_Tribas, ELM_Hardlim,
    QELM_Sigmoid, QELM_Tanh, QELM_ReLU, QELM_Sine, QELM_Tribas, QELM_Hardlim
)
from utils.data_loader import load_dataset
from utils.metrics import compute_metrics
import pandas as pd


def run_benchmark_experiment(datasets=['iris', 'wine', 'breast_cancer', 'digits'], 
                             n_hidden=100, 
                             random_state=42):
    """
    Run benchmark comparison across multiple datasets.
    
    Parameters
    ----------
    datasets : list of str
        List of dataset names to evaluate
    n_hidden : int
        Number of hidden neurons
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    results_df : pandas.DataFrame
        Results table with model performance on each dataset
    """
    # Define all models
    models = {
        'ELM-Sigmoid': ELM_Sigmoid(n_hidden=n_hidden, random_state=random_state),
        'ELM-Tanh': ELM_Tanh(n_hidden=n_hidden, random_state=random_state),
        'ELM-ReLU': ELM_ReLU(n_hidden=n_hidden, random_state=random_state),
        'ELM-Sine': ELM_Sine(n_hidden=n_hidden, random_state=random_state),
        'ELM-Tribas': ELM_Tribas(n_hidden=n_hidden, random_state=random_state),
        'ELM-Hardlim': ELM_Hardlim(n_hidden=n_hidden, random_state=random_state),
        'QELM-Sigmoid': QELM_Sigmoid(n_hidden=n_hidden, random_state=random_state),
        'QELM-Tanh': QELM_Tanh(n_hidden=n_hidden, random_state=random_state),
        'QELM-ReLU': QELM_ReLU(n_hidden=n_hidden, random_state=random_state),
        'QELM-Sine': QELM_Sine(n_hidden=n_hidden, random_state=random_state),
        'QELM-Tribas': QELM_Tribas(n_hidden=n_hidden, random_state=random_state),
        'QELM-Hardlim': QELM_Hardlim(n_hidden=n_hidden, random_state=random_state),
    }
    
    results = []
    
    print("="*80)
    print("EXPERIMENT 1: BENCHMARK COMPARISON")
    print("="*80)
    print(f"Hidden neurons: {n_hidden}")
    print(f"Random seed: {random_state}")
    print(f"Datasets: {', '.join(datasets)}")
    print("="*80)
    
    for dataset_name in datasets:
        print(f"\nEvaluating on {dataset_name} dataset...")
        
        # Load dataset
        X_train, X_test, y_train, y_test, info = load_dataset(dataset_name, random_state=random_state)
        
        print(f"  Dataset info: {info['n_train']} train, {info['n_test']} test, "
              f"{info['n_features']} features, {info['n_classes']} classes")
        
        for model_name, model in models.items():
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Round predictions for classification
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = np.round(y_pred).astype(int)
            
            metrics = compute_metrics(y_test, y_pred)
            
            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })
            
            print(f"  {model_name:15s} - Accuracy: {metrics['accuracy']:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY: Average Performance Across All Datasets")
    print("="*80)
    summary = results_df.groupby('Model')['Accuracy'].agg(['mean', 'std'])
    summary = summary.sort_values('mean', ascending=False)
    print(summary.to_string())
    
    return results_df


def main():
    """Main function to run the experiment."""
    # Run experiment
    results_df = run_benchmark_experiment(
        datasets=['iris', 'wine', 'breast_cancer', 'digits'],
        n_hidden=100,
        random_state=42
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'experiment_1_benchmark_{timestamp}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Create pivot table for paper (Table 1 format)
    pivot = results_df.pivot_table(
        index='Model', 
        columns='Dataset', 
        values='Accuracy'
    )
    pivot_path = os.path.join(output_dir, f'experiment_1_table1_{timestamp}.csv')
    pivot.to_csv(pivot_path)
    print(f"✓ Table 1 format saved to: {pivot_path}")
    
    return results_df


if __name__ == '__main__':
    main()
