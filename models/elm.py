"""
Traditional Extreme Learning Machine (ELM) implementations with various activation functions.
"""
import numpy as np
from .base import BaseELM, Sigmoid, Tanh, ReLU, Sine, Tribas, Hardlim


class TraditionalELM(BaseELM):
    """
    Traditional ELM with random initialization.
    
    This implementation follows the original ELM algorithm where input weights
    and biases are randomly initialized from a uniform distribution.
    """
    
    def _initialize_weights(self, X):
        """Initialize input weights and biases randomly."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Random initialization from uniform distribution [-1, 1]
        self.input_weights = np.random.uniform(-1, 1, (self.n_features, self.n_hidden))
        self.biases = np.random.uniform(-1, 1, (1, self.n_hidden))


class ELM_Sigmoid(TraditionalELM):
    """
    Traditional ELM with sigmoid activation function.
    
    Corresponds to baseline model in Table 1, Row 1 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    random_state : int, optional
        Random seed for reproducibility
        
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> model = ELM_Sigmoid(n_hidden=50, random_state=42)
    >>> model.fit(X[:100], y[:100])
    >>> accuracy = model.score(X[100:], y[100:])
    """
    
    def __init__(self, n_hidden=100, random_state=None):
        super().__init__(n_hidden, Sigmoid(), random_state)


class ELM_Tanh(TraditionalELM):
    """
    Traditional ELM with hyperbolic tangent activation function.
    
    Corresponds to baseline model in Table 1, Row 2 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden=100, random_state=None):
        super().__init__(n_hidden, Tanh(), random_state)


class ELM_ReLU(TraditionalELM):
    """
    Traditional ELM with ReLU activation function.
    
    Corresponds to baseline model in Table 1, Row 3 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden=100, random_state=None):
        super().__init__(n_hidden, ReLU(), random_state)


class ELM_Sine(TraditionalELM):
    """
    Traditional ELM with sine activation function.
    
    Corresponds to baseline model in Table 1, Row 4 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden=100, random_state=None):
        super().__init__(n_hidden, Sine(), random_state)


class ELM_Tribas(TraditionalELM):
    """
    Traditional ELM with triangular basis activation function.
    
    Corresponds to baseline model in Table 1, Row 5 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden=100, random_state=None):
        super().__init__(n_hidden, Tribas(), random_state)


class ELM_Hardlim(TraditionalELM):
    """
    Traditional ELM with hard limit activation function.
    
    Corresponds to baseline model in Table 1, Row 6 of the paper.
    
    Parameters
    ----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden=100, random_state=None):
        super().__init__(n_hidden, Hardlim(), random_state)
