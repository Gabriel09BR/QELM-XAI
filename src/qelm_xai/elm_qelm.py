import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from .activations import _activation


# ===== Classifier =====
class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden=100, generator="new", activation="relu", C=1.0,
                 random_state=None, weight_scale=1.0, use_bias=True,
                 activation_params=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.C = C
        self.random_state = random_state
        self.weight_scale = weight_scale
        self.use_bias = use_bias
        self.generator = generator
        self.activation_params = activation_params if activation_params is not None else {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)  # Random number generator
        n_features = X.shape[1]  # Number of features (columns) in the dataset

        # Learns the existing classes in y and converts labels to binary representation (0/1)
        # Example: [0, 0, 0, 1, 0, 0]
        self.lb_ = LabelBinarizer()
        Y = self.lb_.fit_transform(y)
        self.classes_ = self.lb_.classes_  # Stores learned classes (sorted)
        self.lb_ = LabelBinarizer()

        # Binary case: force 2 columns aligned with class order
        if Y.ndim == 1:  # 0/1 vector
            Y = np.column_stack([1 - Y, Y]).astype(float)

        self.n_outputs_ = Y.shape[1]

        if self.generator == "new":
            # Random-Forest-like generator
            # For each neuron, choose one feature and one threshold
            feat_idx = rng.integers(0, n_features, size=self.n_hidden)

            # Robust thresholds: sample quantiles per neuron (avoids outliers)
            q_low = float(self.activation_params.get("q_low", 0.1))
            q_high = float(self.activation_params.get("q_high", 0.9))

            # Returns n_hidden random values between q_low and q_high
            quantiles = np.clip(
                rng.uniform(q_low, q_high, size=self.n_hidden),
                0.0, 1.0
            )

            thresholds = np.array(
                [np.quantile(X[:, j], q) for j, q in zip(feat_idx, quantiles)],
                dtype=float
            )

            # Random polarity:
            # +1 => test (X_j - t) >= 0  ==> X_j >= t
            # -1 => test (t - X_j) >= 0  ==> X_j <= t
            polarities = rng.choice([-1.0, 1.0], size=self.n_hidden)

            # One-hot W matrix with polarity; bias adjusted to the threshold
            self.W_ = np.zeros((n_features, self.n_hidden), dtype=float)
            cols = np.arange(self.n_hidden)
            self.W_[feat_idx, cols] = polarities
            self.b_ = (-polarities * thresholds).astype(float)

            # Data stored for inspection / explainability
            self.rf_feat_idx_ = feat_idx
            self.rf_thresholds_ = thresholds
            self.rf_polarities_ = polarities

            act_kind = self.activation

        else:
            # Huang's uniform random generator
            self.W_ = rng.uniform(-1.0, 1.0, size=(n_features, self.n_hidden))
            self.b_ = (
                rng.uniform(-1.0, 1.0, size=(self.n_hidden,))
                if self.use_bias
                else np.zeros(self.n_hidden)
            )

            act_kind = self.activation

        # Hidden layer matrix H
        A = X @ self.W_ + self.b_
        H = _activation(A, act_kind, **self.activation_params)

        # Ridge-ELM (closed-form solution)
        lam = 0.0 if self.C is None or np.isinf(self.C) else 1.0 / self.C
        G = H.T @ H
        if lam > 0:
            G += lam * np.eye(G.shape[0])

        self.beta_ = np.linalg.solve(G, H.T @ Y)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        A = X @ self.W_ + self.b_
        # Uses the same activation kind as in fit
        kind = self.activation
        H = _activation(A, kind, **self.activation_params)
        return H @ self.beta_

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Z = self.decision_function(X).astype(float)

        if Z.ndim == 1:
            Z = Z[:, None]

        if Z.shape[1] == 1 and len(self.classes_) == 2:
            # Convert score to probability using logistic function
            # and complete to two columns
            p1 = 1.0 / (1.0 + np.exp(-Z.ravel()))
            P = np.column_stack([1.0 - p1, p1])
        else:
            # Softmax for multiclass
            Z -= Z.max(axis=1, keepdims=True)
            Z = np.clip(Z, -80.0, 80.0)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)

        return P

    def predict(self, X):
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        return self.classes_[idx]


