"""
Base classes and activation functions for ELM and QELM models.
"""
import numpy as np
from abc import ABC, abstractmethod


class ActivationFunction:
    """Base class for activation functions."""
    
    @staticmethod
    @abstractmethod
    def activate(x):
        """Apply activation function to input."""
        pass
    
    @staticmethod
    @abstractmethod
    def name():
        """Return the name of the activation function."""
        pass


class Sigmoid(ActivationFunction):
    """Sigmoid activation function: σ(x) = 1 / (1 + exp(-x))"""
    
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    
    @staticmethod
    def name():
        return "sigmoid"


class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function: tanh(x)"""
    
    @staticmethod
    def activate(x):
        return np.tanh(x)
    
    @staticmethod
    def name():
        return "tanh"


class ReLU(ActivationFunction):
    """Rectified Linear Unit: ReLU(x) = max(0, x)"""
    
    @staticmethod
    def activate(x):
        return np.maximum(0, x)
    
    @staticmethod
    def name():
        return "relu"


class Sine(ActivationFunction):
    """Sine activation function: sin(x)"""
    
    @staticmethod
    def activate(x):
        return np.sin(x)
    
    @staticmethod
    def name():
        return "sine"


class Tribas(ActivationFunction):
    """Triangular basis function: max(0, 1 - |x|)"""
    
    @staticmethod
    def activate(x):
        return np.maximum(0, 1 - np.abs(x))
    
    @staticmethod
    def name():
        return "tribas"


class Hardlim(ActivationFunction):
    """Hard limit activation: 1 if x >= 0, else 0"""
    
    @staticmethod
    def activate(x):
        return (x >= 0).astype(float)
    
    @staticmethod
    def name():
        return "hardlim"


class BaseELM(ABC):
    """
    Base class for Extreme Learning Machine models.
    
    Parameters
    ----------
    n_hidden : int
        Number of hidden layer neurons
    activation : ActivationFunction
        Activation function to use in the hidden layer
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden, activation, random_state=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state
        self.input_weights = None
        self.biases = None
        self.output_weights = None
        self.n_features = None
        self.n_outputs = None
        
    @abstractmethod
    def _initialize_weights(self, X):
        """Initialize input weights and biases."""
        pass
    
    def _compute_hidden_output(self, X):
        """Compute hidden layer output H."""
        G = np.dot(X, self.input_weights) + self.biases
        H = self.activation.activate(G)
        return H
    
    def fit(self, X, y):
        """
        Train the ELM model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values
            
        Returns
        -------
        self : object
        """
        X = np.array(X)
        y = np.array(y)
        
        # Handle 1D target
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]
        
        # Initialize weights
        self._initialize_weights(X)
        
        # Compute hidden layer output
        H = self._compute_hidden_output(X)
        
        # Calculate output weights using Moore-Penrose pseudoinverse
        self.output_weights = np.dot(np.linalg.pinv(H), y)
        
        return self
    
    def predict(self, X):
        """
        Predict using the ELM model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred : array, shape (n_samples,) or (n_samples, n_outputs)
            Predicted values
        """
        X = np.array(X)
        H = self._compute_hidden_output(X)
        y_pred = np.dot(H, self.output_weights)
        
        # Return 1D array if single output
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred
    
    def score(self, X, y):
        """
        Calculate accuracy for classification tasks.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test data
        y : array-like, shape (n_samples,)
            True labels
            
        Returns
        -------
        score : float
            Accuracy score
        """
        y_pred = self.predict(X)
        
        # For multi-output, take argmax
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            # Round for binary classification
            y_pred = np.round(y_pred)
        
        return np.mean(y_pred == y)
