"""
Run all experiments from the paper.
This script executes all experiments in sequence and generates a complete report.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
import experiment_1_benchmark
import experiment_2_robustness
import experiment_3_activation


def main():
    """Run all experiments."""
    print("="*80)
    print("RUNNING ALL EXPERIMENTS FROM THE PAPER")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Experiment 1: Benchmark comparison
    print("\n\n" + "="*80)
    print("STARTING EXPERIMENT 1: BENCHMARK COMPARISON")
    print("="*80 + "\n")
    try:
        exp1_results = experiment_1_benchmark.main()
        print("\n✓ Experiment 1 completed successfully!")
    except Exception as e:
        print(f"\n✗ Experiment 1 failed with error: {e}")
    
    # Experiment 2: Robustness analysis
    print("\n\n" + "="*80)
    print("STARTING EXPERIMENT 2: ROBUSTNESS ANALYSIS")
    print("="*80 + "\n")
    try:
        exp2_results = experiment_2_robustness.main()
        print("\n✓ Experiment 2 completed successfully!")
    except Exception as e:
        print(f"\n✗ Experiment 2 failed with error: {e}")
    
    # Experiment 3: Activation function comparison
    print("\n\n" + "="*80)
    print("STARTING EXPERIMENT 3: ACTIVATION FUNCTION COMPARISON")
    print("="*80 + "\n")
    try:
        exp3_results = experiment_3_activation.main()
        print("\n✓ Experiment 3 completed successfully!")
    except Exception as e:
        print(f"\n✗ Experiment 3 failed with error: {e}")
    
    # Summary
    print("\n\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults have been saved to the 'results/' directory.")
    print("Please refer to the individual CSV files for detailed analysis.")
    print("="*80)


if __name__ == '__main__':
    main()
