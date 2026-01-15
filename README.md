# QELM-XAI: Extreme Learning Machine with Quantile-Based Variants and Explainability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository accompanies the scientific article on **Quantile-based Extreme Learning Machine (QELM)** variants with multiple activation functions. The project provides reproducible implementations to validate the research findings presented in the paper.

## 📖 Research Motivation

Extreme Learning Machines (ELM) are single-hidden-layer feedforward neural networks with randomly initialized hidden layer weights and analytically computed output weights. While ELMs are known for their fast training speed and good generalization performance, their sensitivity to outliers and initialization can limit performance in certain scenarios.

This research introduces **Quantile-based ELM (QELM)** variants that:
- **Improve robustness** to outliers through quantile-based weight initialization
- **Enhance stability** across different random seeds
- **Maintain computational efficiency** of traditional ELMs
- **Support multiple activation functions** (sigmoid, tanh, ReLU, sine, tribas, hardlim)

## 🎯 Key Features

- **Explicit Model Naming**: All model classes use clear, descriptive names matching tables and figures in the paper
- **Multiple Activation Functions**: Support for sigmoid, tanh, ReLU, sine, triangular basis (tribas), and hard limit (hardlim)
- **Comprehensive Evaluation**: Performance metrics, statistical tests, and visualization tools
- **Well-Documented**: Inline comments and docstrings for all implementations

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Gabriel09BR/QELM-XAI.git
cd QELM-XAI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from models.elm import ELM_Sigmoid, ELM_Tanh, ELM_ReLU
from models.qelm import QELM_Sigmoid, QELM_Tanh, QELM_ReLU
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train traditional ELM with sigmoid activation
elm = ELM_Sigmoid(n_hidden=100, random_state=42)
elm.fit(X_train, y_train)
elm_score = elm.score(X_test, y_test)
print(f"ELM-Sigmoid Accuracy: {elm_score:.4f}")

# Train QELM with sigmoid activation
qelm = QELM_Sigmoid(n_hidden=100, quantile=0.5, random_state=42)
qelm.fit(X_train, y_train)
qelm_score = qelm.score(X_test, y_test)
print(f"QELM-Sigmoid Accuracy: {qelm_score:.4f}")
```

## 📁 Repository Structure

```
QELM-XAI/
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── elm.py                # Traditional ELM variants
│   ├── qelm.py               # Quantile-based ELM variants
│   └── base.py               # Base classes and activation functions
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── data_loader.py       # Dataset loading and preprocessing
│   ├── metrics.py           # Evaluation metrics
│   └── visualization.py     # Plotting and visualization
├── notebooks/                # Jupyter notebooks for analysis
├── data/                     # Datasets (see data/README.md)
│   └── README.md
├── results/                  # Results and outputs
│   └── README.md
├── requirements.txt          # Python dependencies
├── example.py               # Quick start example
├── README.md                # This file
├── LICENSE                  # License information
└── CITATION.cff             # Citation information
```

## 📊 Model Naming Convention

To ensure clarity and reproducibility, all models follow an explicit naming convention:

| Model Class | Description | Reference in Paper |
|------------|-------------|-------------------|
| `ELM_Sigmoid` | Traditional ELM with sigmoid activation | Baseline (Table 1, Row 1) |
| `ELM_Tanh` | Traditional ELM with tanh activation | Baseline (Table 1, Row 2) |
| `ELM_ReLU` | Traditional ELM with ReLU activation | Baseline (Table 1, Row 3) |
| `ELM_Sine` | Traditional ELM with sine activation | Baseline (Table 1, Row 4) |
| `ELM_Tribas` | Traditional ELM with triangular basis | Baseline (Table 1, Row 5) |
| `ELM_Hardlim` | Traditional ELM with hard limit | Baseline (Table 1, Row 6) |
| `QELM_Sigmoid` | QELM with sigmoid activation | Proposed (Table 1, Row 7) |
| `QELM_Tanh` | QELM with tanh activation | Proposed (Table 1, Row 8) |
| `QELM_ReLU` | QELM with ReLU activation | Proposed (Table 1, Row 9) |
| `QELM_Sine` | QELM with sine activation | Proposed (Table 1, Row 10) |
| `QELM_Tribas` | QELM with triangular basis | Proposed (Table 1, Row 11) |
| `QELM_Hardlim` | QELM with hard limit | Proposed (Table 1, Row 12) |

## 🔍 Explainability (XAI)

The repository includes tools for interpreting ELM/QELM models:
- Feature importance analysis
- Hidden layer weight visualization
- Decision boundary plotting (for 2D datasets)

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{qelm2024,
  title={Quantile-based Extreme Learning Machine: Improving Robustness and Performance},
  author={Author Names},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XXX-XXX}
}
```

Or use the `CITATION.cff` file for automatic citation formatting.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the original ELM work by Huang et al.
- Built with scikit-learn for seamless integration with the Python ML ecosystem
- Thanks to the open-source community for valuable feedback

## 📧 Contact

For questions or collaborations, please open an issue or contact the authors through GitHub.

---

**Note**: This repository is designed for academic research and education. The implementations prioritize clarity and reproducibility over computational optimization.
