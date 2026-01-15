"""
Quantile-based Extreme Learning Machine (QELM) implementations.
"""
import numpy as np
from .base import BaseELM, Sigmoid, Tanh, ReLU, Sine, Tribas, Hardlim


class QuantileELM(BaseELM):
    """
    Quantile-based ELM with robust initialization.
    
    This implementation uses quantile-based initialization to improve
    robustness to outliers and initialization sensitivity.
    
    Parameters
    ----------
    quantile : float, default=0.5
        Quantile to use for initialization (0.5 = median)
    """
    
    def __init__(self, n_hidden, activation, quantile=0.5, random_state=None):
        super().__init__(n_hidden, activation, random_state)
        self.quantile = quantile
    
    def _initialize_weights(self, X):
        """
        Initialize weights using quantile-based approach.
        
        This method computes quantiles of the input data to determine
        appropriate scaling for the random weights, improving robustness.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Compute quantile-based scaling factors
        q_lower = np.quantile(X, 1 - self.quantile, axis=0)
        q_upper = np.quantile(X, self.quantile, axis=0)
        scale = np.maximum(q_upper - q_lower, 1e-8)  # Avoid division by zero
        
        # Initialize weights with quantile-based scaling
        self.input_weights = np.random.uniform(-1, 1, (self.n_features, self.n_hidden))
        self.input_weights = self.input_weights * scale.reshape(-1, 1)
        
        # Bias initialization based on data center
        center = (q_lower + q_upper) / 2
        self.biases = np.random.uniform(-1, 1, (1, self.n_hidden)) - np.dot(center.reshape(1, -1), self.input_weights)


class QELM_Sigmoid(QuantileELM):
    """
    Quantile-based ELM with sigmoid activation function.
    
    Corresponds to proposed model in Table 1, Row 7 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    quantile : float, default=0.5
        Quantile to use for weight initialization (0.5 = median-based)
    random_state : int, optional
        Random seed for reproducibility
        
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> model = QELM_Sigmoid(n_hidden=50, quantile=0.5, random_state=42)
    >>> model.fit(X[:100], y[:100])
    >>> accuracy = model.score(X[100:], y[100:])
    """
    
    def __init__(self, n_hidden=100, quantile=0.5, random_state=None):
        super().__init__(n_hidden, Sigmoid(), quantile, random_state)


class QELM_Tanh(QuantileELM):
    """
    Quantile-based ELM with hyperbolic tangent activation function.
    
    Corresponds to proposed model in Table 1, Row 8 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    quantile : float, default=0.5
        Quantile to use for weight initialization
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden=100, quantile=0.5, random_state=None):
        super().__init__(n_hidden, Tanh(), quantile, random_state)


class QELM_ReLU(QuantileELM):
    """
    Quantile-based ELM with ReLU activation function.
    
    Corresponds to proposed model in Table 1, Row 9 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    quantile : float, default=0.5
        Quantile to use for weight initialization
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden=100, quantile=0.5, random_state=None):
        super().__init__(n_hidden, ReLU(), quantile, random_state)


class QELM_Sine(QuantileELM):
    """
    Quantile-based ELM with sine activation function.
    
    Corresponds to proposed model in Table 1, Row 10 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    quantile : float, default=0.5
        Quantile to use for weight initialization
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden=100, quantile=0.5, random_state=None):
        super().__init__(n_hidden, Sine(), quantile, random_state)


class QELM_Tribas(QuantileELM):
    """
    Quantile-based ELM with triangular basis activation function.
    
    Corresponds to proposed model in Table 1, Row 11 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    quantile : float, default=0.5
        Quantile to use for weight initialization
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden=100, quantile=0.5, random_state=None):
        super().__init__(n_hidden, Tribas(), quantile, random_state)


class QELM_Hardlim(QuantileELM):
    """
    Quantile-based ELM with hard limit activation function.
    
    Corresponds to proposed model in Table 1, Row 12 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    quantile : float, default=0.5
        Quantile to use for weight initialization
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden=100, quantile=0.5, random_state=None):
        super().__init__(n_hidden, Hardlim(), quantile, random_state)
