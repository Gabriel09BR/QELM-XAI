"""
Simple Example: Getting Started with ELM and QELM

This script demonstrates basic usage of ELM and QELM models.
Perfect for new users to understand the API.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models import ELM_Sigmoid, QELM_Sigmoid
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    """Run a simple comparison example."""
    print("="*70)
    print("QELM-XAI: Simple Example")
    print("="*70)
    
    # Load the Iris dataset
    print("\n1. Loading Iris dataset...")
    X, y = load_iris(return_X_y=True)
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of classes: {len(set(y))}")
    
    # Split into train and test sets
    print("\n2. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Normalize features
    print("\n3. Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("   Features standardized (mean=0, std=1)")
    
    # Train traditional ELM
    print("\n4. Training traditional ELM with sigmoid activation...")
    elm = ELM_Sigmoid(n_hidden=100, random_state=42)
    elm.fit(X_train, y_train)
    elm_score = elm.score(X_test, y_test)
    print(f"   ELM Test Accuracy: {elm_score:.4f} ({elm_score*100:.2f}%)")
    
    # Train QELM
    print("\n5. Training QELM with sigmoid activation...")
    qelm = QELM_Sigmoid(n_hidden=100, quantile=0.5, random_state=42)
    qelm.fit(X_train, y_train)
    qelm_score = qelm.score(X_test, y_test)
    print(f"   QELM Test Accuracy: {qelm_score:.4f} ({qelm_score*100:.2f}%)")
    
    # Compare
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"ELM Accuracy:  {elm_score:.4f}")
    print(f"QELM Accuracy: {qelm_score:.4f}")
    improvement = (qelm_score - elm_score) * 100
    print(f"Improvement:   {improvement:+.2f}%")
    print("="*70)
    
    print("\n✓ Example completed successfully!")
    print("\nNext steps:")
    print("  - Try different activation functions (ELM_Tanh, ELM_ReLU, etc.)")
    print("  - Experiment with different n_hidden values")
    print("  - Check out the notebooks/ directory for more examples")
    print("  - Explore the utility functions in utils/ for data loading and visualization")


if __name__ == '__main__':
    main()
