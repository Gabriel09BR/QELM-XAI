import numpy as np

# ===== Activations =====
def _activation(X, kind, **params):
    if kind == "relu":
        return np.maximum(0, X)
    elif kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-X))
    elif kind == "tanh":
        return np.tanh(X)
    elif kind == "linear":
        return X
    elif kind == "sine":
        return np.sin(X)
    elif kind == "hardlim":
        return (X >= 0).astype(np.float64)
    elif kind == "tribas":
        return np.maximum(1 - np.abs(X), 0)
    elif kind == "poly":
        deg   = int(params.get("degree", 2))
        gamma = float(params.get("gamma", 1.0))
        coef0 = float(params.get("coef0", 0.0))
        Z = gamma * X + coef0
        return np.power(Z, deg, dtype=np.float64)
    elif kind == "radbas":
        gamma = float(params.get("gamma", 1.0))
        if "sigma" in params:
            sigma = float(params["sigma"])
            gamma = 1.0/(sigma**2 + 1e-12) / 2.0
        return np.exp(-gamma * np.square(X))
    else:
        raise ValueError(f"Activation '{kind}' not supported.")

