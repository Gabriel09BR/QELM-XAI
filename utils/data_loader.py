"""
Data loading and preprocessing utilities.
"""
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
    make_classification
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


def load_dataset(name, test_size=0.3, random_state=42, normalize=True):
    """
    Load a standard benchmark dataset.
    
    Parameters
    ----------
    name : str
        Dataset name: 'iris', 'wine', 'breast_cancer', 'digits', or 'synthetic'
    test_size : float, default=0.3
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random seed for reproducibility
    normalize : bool, default=True
        Whether to normalize features using StandardScaler
        
    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Training and test splits
    dataset_info : dict
        Information about the dataset
    """
    # Load dataset
    if name == 'iris':
        X, y = load_iris(return_X_y=True)
        n_classes = 3
    elif name == 'wine':
        X, y = load_wine(return_X_y=True)
        n_classes = 3
    elif name == 'breast_cancer':
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = 2
    elif name == 'digits':
        X, y = load_digits(return_X_y=True)
        n_classes = 10
    elif name == 'synthetic':
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=random_state
        )
        n_classes = 3
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    dataset_info = {
        'name': name,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': n_classes,
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0]
    }
    
    return X_train, X_test, y_train, y_test, dataset_info


def add_outliers(X, y, outlier_fraction=0.1, noise_factor=3.0, random_state=42):
    """
    Add outliers to a dataset for robustness testing.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data
    y : array-like, shape (n_samples,)
        Target labels
    outlier_fraction : float, default=0.1
        Fraction of samples to convert to outliers
    noise_factor : float, default=3.0
        Multiplier for noise magnitude
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    X_noisy, y_noisy : arrays
        Data with added outliers
    """
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    n_outliers = int(outlier_fraction * n_samples)
    
    X_noisy = X.copy()
    y_noisy = y.copy()
    
    # Select random samples to make into outliers
    outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
    
    # Add noise to features
    noise = rng.randn(n_outliers, X.shape[1]) * noise_factor * X.std(axis=0)
    X_noisy[outlier_indices] += noise
    
    return X_noisy, y_noisy
