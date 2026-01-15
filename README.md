# QELM-XAI: Quantile-based Extreme Learning Machine for Explainable AI

This repository contains the reference implementation used in the study of **Quantile-based Extreme Learning Machines (QELM)** with a focus on **Explainable Artificial Intelligence (XAI)**.

The project provides multiple ELM and QELM model variants with explicit and human-readable model names, enabling clear identification of each configuration in experiments, logs, tables, and explainability analyses.

---

## Research Motivation

Extreme Learning Machines (ELMs) are known for their fast training and low computational cost, making them attractive for real-world and resource-constrained applications.  
Quantile-based ELMs (QELMs) extend this paradigm by incorporating quantile-based decision mechanisms, improving robustness and interpretability.

In explainable AI studies, a major challenge lies in **maintaining a transparent mapping between model implementations and reported results**. To address this, each ELM and QELM variant in this project is represented by a distinct model class (e.g., `ELMRelu`, `QELMSigmoid`, `QELMRadbas`). This design choice is essential for XAI pipelines, where explanations must be explicitly associated with the correct model configuration.

---

## Main Contributions

- Implementation of classical **ELM** and **Quantile-based ELM (QELM)** classifiers
- Explicit model naming to ensure traceability in XAI analyses
- Support for multiple activation functions
- Lightweight and efficient models suitable for explainability studies
- Clean structure designed for reproducible scientific experiments

---

## Explainability Perspective (QELM-XAI)

This repository was designed to support explainability workflows, including:
- Model-level comparison between ELM and QELM variants
- Transparent association between explanations and model configurations
- Use in post-hoc XAI methods such as feature attribution and sensitivity analysis
- Clear reporting of explainability results in scientific articles

Explicit model naming avoids ambiguity when generating explanations, plots, and tables, which is critical in XAI-oriented research.

---

## Project Structure

```text
.
├── src/                # Core model implementations
│   ├── elm.py          # Base ELM / QELM classifier
│   └── variants.py     # Explicitly named ELM and QELM variants
├── experiments/        # Experimental scripts
├── results/            # Metrics, tables, and outputs
├── figures/            # Visualizations and explainability plots
├── requirements.txt    # Python dependencies
└── README.md
