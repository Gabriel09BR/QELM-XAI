"""
Experiment 3: Activation Function Comparison
This experiment compares different activation functions for ELM and QELM.
Results correspond to Figure 2 in the paper.
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
import pandas as pd


def run_activation_comparison(dataset='iris', n_hidden=100, random_state=42):
    """
    Compare different activation functions.
    
    Parameters
    ----------
    dataset : str
        Dataset name to use
    n_hidden : int
        Number of hidden neurons
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    results_df : pandas.DataFrame
        Comparison results
    """
    # Define model configurations
    activations = ['Sigmoid', 'Tanh', 'ReLU', 'Sine', 'Tribas', 'Hardlim']
    
    elm_models = {
        'Sigmoid': ELM_Sigmoid(n_hidden=n_hidden, random_state=random_state),
        'Tanh': ELM_Tanh(n_hidden=n_hidden, random_state=random_state),
        'ReLU': ELM_ReLU(n_hidden=n_hidden, random_state=random_state),
        'Sine': ELM_Sine(n_hidden=n_hidden, random_state=random_state),
        'Tribas': ELM_Tribas(n_hidden=n_hidden, random_state=random_state),
        'Hardlim': ELM_Hardlim(n_hidden=n_hidden, random_state=random_state),
    }
    
    qelm_models = {
        'Sigmoid': QELM_Sigmoid(n_hidden=n_hidden, random_state=random_state),
        'Tanh': QELM_Tanh(n_hidden=n_hidden, random_state=random_state),
        'ReLU': QELM_ReLU(n_hidden=n_hidden, random_state=random_state),
        'Sine': QELM_Sine(n_hidden=n_hidden, random_state=random_state),
        'Tribas': QELM_Tribas(n_hidden=n_hidden, random_state=random_state),
        'Hardlim': QELM_Hardlim(n_hidden=n_hidden, random_state=random_state),
    }
    
    results = []
    
    print("="*80)
    print("EXPERIMENT 3: ACTIVATION FUNCTION COMPARISON")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Hidden neurons: {n_hidden}")
    print(f"Random seed: {random_state}")
    print("="*80)
    
    # Load dataset
    X_train, X_test, y_train, y_test, info = load_dataset(dataset, random_state=random_state)
    print(f"\nDataset info: {info['n_train']} train, {info['n_test']} test")
    
    # Evaluate ELM models
    print("\nTraditional ELM:")
    for activation in activations:
        model = elm_models[activation]
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        results.append({
            'Activation': activation,
            'Model_Type': 'ELM',
            'Accuracy': accuracy
        })
        
        print(f"  {activation:10s}: {accuracy:.4f}")
    
    # Evaluate QELM models
    print("\nQuantile-based ELM:")
    for activation in activations:
        model = qelm_models[activation]
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        results.append({
            'Activation': activation,
            'Model_Type': 'QELM',
            'Accuracy': accuracy
        })
        
        print(f"  {activation:10s}: {accuracy:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Display comparison
    print("\n" + "="*80)
    print("ACTIVATION FUNCTION RANKING")
    print("="*80)
    
    pivot = results_df.pivot(index='Activation', columns='Model_Type', values='Accuracy')
    pivot['Improvement'] = pivot['QELM'] - pivot['ELM']
    pivot = pivot.sort_values('QELM', ascending=False)
    
    print(pivot.to_string())
    
    return results_df


def main():
    """Main function to run the experiment."""
    # Run experiment on multiple datasets
    all_results = []
    
    for dataset in ['iris', 'wine', 'breast_cancer']:
        print(f"\n{'='*80}")
        print(f"Testing on {dataset.upper()} dataset")
        print('='*80)
        
        results_df = run_activation_comparison(
            dataset=dataset,
            n_hidden=100,
            random_state=42
        )
        results_df['Dataset'] = dataset
        all_results.append(results_df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'experiment_3_activation_{timestamp}.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    return combined_df


if __name__ == '__main__':
    main()
