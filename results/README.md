# Results Directory

This directory stores experiment outputs and visualizations.

## Generated Files

Experiment scripts automatically save results here with timestamps:

- `experiment_1_benchmark_YYYYMMDD_HHMMSS.csv` - Benchmark comparison results (Table 1)
- `experiment_1_table1_YYYYMMDD_HHMMSS.csv` - Table 1 formatted for paper
- `experiment_2_robustness_YYYYMMDD_HHMMSS.csv` - Robustness analysis results (Table 2)
- `experiment_3_activation_YYYYMMDD_HHMMSS.csv` - Activation function comparison (Figure 2)

## Plots and Figures

Visualization scripts may also save figures here:
- PNG or PDF format
- 300 DPI for publication quality

## Note

Result files are excluded from version control to keep the repository clean.
You can regenerate all results by running the experiment scripts.

## Reproducing Paper Results

To reproduce the exact results from the paper:

```bash
cd experiments
python run_all_experiments.py
```

This will generate all tables and figures referenced in the manuscript.
