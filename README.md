# Max K-Cut QUBO Reformulations

Characterization of QUBO reformulations for the max k-cut problem using QAOA.

## Overview

This package implements and compares four different optimization formulations for the Max K-Cut problem:
- **BQO** (Binary Quadratic Optimization): Full formulation with explicit constraints
- **QUBO** (Quadratic Unconstrained Binary Optimization): BQO with penalty terms
- **RBQO** (Reduced BQO): Reduced variable formulation (K-1 variables per node)
- **RQUBO** (Reduced QUBO): RBQO with penalty terms

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from max_k_cut import generate_graph, docplex_QUBO, tight_qubo_penalty
from max_k_cut.qaoa import run_qaoa_extract_samples
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit.primitives import Sampler

# Generate a random graph
G = generate_graph(num_nodes=6, edge_probability=0.5, weighted=True, weight_range=10)

# Create QUBO model with tight penalty
K = 3
penalty = tight_qubo_penalty(G, K)
model = docplex_QUBO(G, K, penalty, name="Max-K-Cut")

# Set up QAOA
sampler = Sampler()
qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=3)
qaoa_optimizer = MinimumEigenOptimizer(qaoa)

# Run QAOA
results = run_qaoa_extract_samples([model], ["QUBO"], optimizer=qaoa_optimizer)
samples = results["QUBO"]["samples"]
```

## Package Structure

```
Max_K_Cut/
├── max_k_cut/              # Main package directory
│   ├── __init__.py
│   ├── models.py           # Model formulations (BQO, QUBO, RBQO, RQUBO)
│   ├── penalties.py         # Penalty functions (tight, naive, interpolated)
│   ├── helpers.py           # Utility functions (graph generation, filtering, etc.)
│   └── qaoa.py              # QAOA execution and sample extraction
├── notebooks/
│   ├── experiments/         # Experiment notebooks
│   │   ├── generate_results.ipynb
│   │   └── penalty_interpolation.ipynb
│   └── analysis/            # Analysis and visualization notebooks
│       ├── visualize_results.ipynb
│       ├── box_plots.ipynb
│       ├── model_testing.ipynb
│       └── infeasible_analysis.ipynb
├── scripts/                 # Standalone utility scripts
│   └── plot_histogram.py
├── examples/                # Example scripts
│   └── basic_usage.py
├── data/                    # Data directory (gitignored)
│   └── results/             # Experimental results (.npz files)
├── tests/                   # Unit tests (optional)
│   └── test_models.py
├── README.md
├── MIGRATION_GUIDE.md       # Guide for updating old code
├── REPACKAGING_PLAN.md      # Original repackaging plan
└── requirements.txt
```

## Key Components

### Models (`models.py`)
- `docplex_BQO()`: Binary Quadratic Optimization model
- `docplex_QUBO()`: QUBO formulation with penalties
- `docplex_RBQO()`: Reduced BQO model
- `docplex_RQUBO()`: Reduced QUBO model

### Penalties (`penalties.py`)
- `tight_qubo_penalty()`: Theoretically tight penalty values
- `naive_qubo_penalty()`: Simple heuristic penalties
- `interpolated_qubo_penalty()`: Interpolation between tight and naive
- Similar functions for RQUBO

### QAOA Execution (`qaoa.py`)
- `run_qaoa_extract_samples()`: Run QAOA and extract samples with approximation ratios

### Helper Functions (`helpers.py`)
- `generate_graph()`: Generate random Erdos-Renyi graphs
- `feasibility_filter()`: Filter infeasible solutions
- `compute_bqo_objective()`: Compute objective values
- Solution conversion functions

## Running Experiments

### Main Experiment Notebook
The primary experiment is in `notebooks/experiments/generate_results.ipynb`:
- Generates random graphs
- Tests QUBO and RQUBO with interpolated penalties
- Runs QAOA with noisy simulation
- Saves results to `data/results/`

### Visualization
Load and visualize results using `notebooks/analysis/visualize_results.ipynb`:
- Box plots comparing QUBO vs RQUBO
- Feasibility analysis
- Approximation ratio distributions

## Citation

If you use this code, please cite:
```bibtex
@misc{harkness2025characterizingquboreformulationsmaxkcut,
      title={Characterizing QUBO Reformulations of the Max-k-Cut Problem for Quantum Computing}, 
      author={Adrian Harkness and Hamidreza Validi and Ramin Fakhimi and Illya V. Hicks and Tamás Terlaky and Luis F. Zuluaga},
      year={2025},
      eprint={2511.01108},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2511.01108}, 
}
```

## License

See LICENSE file for details.
