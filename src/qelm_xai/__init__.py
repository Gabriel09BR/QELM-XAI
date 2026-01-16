#Python version in use is 3.11.13

!pip install pandas
!pip install numpy==1.26.4 scipy==1.11.4 scikit-learn==1.4.2 joblib==1.4.2
!pip install pycaret==3.3.2 shap==0.46.0

# ===== Bibliotecas padrão =====
import re
import struct
from copy import deepcopy
from itertools import product
from time import time, perf_counter

# ===== Bibliotecas científicas =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ===== IPython =====
from IPython.display import clear_output

# ===== Scikit-learn =====
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# ===== PyCaret (classification) =====
from pycaret.classification import (
    setup,
    create_model,
    tune_model,
    finalize_model,
    predict_model,
    pull,
    get_config
)

# ===== PyCaret utils =====
from pycaret.utils import *

# =========================================================
# Core model
# =========================================================

from .models.elm import ELMClassifier

# =========================================================
# Named model aliases (for logging / experiments / papers)
# =========================================================

from .models.aliases import (
    # ELM variants
    ELMRelu,
    ELMSigmoid,
    ELMTribas,
    ELMRadbas,
    ELMHardlim,
    ELMLinear,
    ELMSine,
    ELMTan,
    ELMPoly,

    # QELM variants
    QELMRelu,
    QELMSigmoid,
    QELMTribas,
    QELMRadbas,
    QELMHardlim,
    QELMLinear,
    QELMSine,
    QELMTan,
    QELMPoly,
)


