# Data Directory

This directory is intended for storing datasets used in experiments.

## Organization

- `example/` - Small example datasets for quick testing
- Other dataset files can be placed here

## Note

Large dataset files are excluded from version control (see `.gitignore`). 
The experiment scripts use scikit-learn's built-in datasets by default:
- Iris
- Wine
- Breast Cancer
- Digits

## Adding Custom Datasets

To use custom datasets:

1. Place your data files in this directory
2. Update `utils/data_loader.py` to include loading functions for your datasets
3. Ensure proper preprocessing (normalization, train/test split, etc.)

## Supported Formats

- CSV files
- NumPy arrays (.npy, .npz)
- Standard scikit-learn datasets
