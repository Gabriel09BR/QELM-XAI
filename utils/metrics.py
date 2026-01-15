"""
Evaluation metrics and statistical tests.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(y_true, y_pred, average='weighted'):
    """
    Compute classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    average : str, default='weighted'
        Averaging method for multi-class metrics
        
    Returns
    -------
    metrics : dict
        Dictionary containing accuracy, precision, recall, and F1 score
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics


def cross_validation_score(model_class, X, y, n_splits=5, random_state=42, **model_params):
    """
    Perform cross-validation and return scores.
    
    Parameters
    ----------
    model_class : class
        Model class to instantiate
    X : array-like
        Input features
    y : array-like
        Target labels
    n_splits : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random seed for reproducibility
    **model_params : dict
        Parameters to pass to model constructor
        
    Returns
    -------
    scores : array
        Array of accuracy scores for each fold
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.array(scores)


def stability_analysis(model_class, X_train, X_test, y_train, y_test, n_runs=30, **model_params):
    """
    Analyze model stability across multiple random initializations.
    
    Parameters
    ----------
    model_class : class
        Model class to test
    X_train, X_test : arrays
        Training and test data
    y_train, y_test : arrays
        Training and test labels
    n_runs : int, default=30
        Number of runs with different random seeds
    **model_params : dict
        Parameters to pass to model (except random_state)
        
    Returns
    -------
    results : dict
        Dictionary with mean, std, min, max accuracy
    """
    scores = []
    
    for seed in range(n_runs):
        params = model_params.copy()
        params['random_state'] = seed
        
        model = model_class(**params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    scores = np.array(scores)
    
    results = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'scores': scores
    }
    
    return results
