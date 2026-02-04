"""
Basic usage example for Max K-Cut QUBO reformulations.

This example demonstrates how to:
1. Generate a random graph
2. Create QUBO and RQUBO models
3. Run QAOA optimization
4. Analyze results
"""

import sys
import os

# Add parent directory to path to import max_k_cut
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from max_k_cut import (
    generate_graph,
    docplex_QUBO,
    docplex_RQUBO,
    tight_qubo_penalty,
    tight_rqubo_penalty,
    run_qaoa_extract_samples,
    feasibility_filter,
    expected_value,
)

from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

import numpy as np

def main():
    # Parameters
    num_nodes = 6
    K = 3
    edge_probability = 0.5
    weighted = True
    weight_range = 10
    
    # Generate a random graph
    print("Generating random graph...")
    G = generate_graph(num_nodes, edge_probability, weighted, weight_range, seed=42)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Create penalty functions
    print("\nComputing penalty functions...")
    penalty_qubo = tight_qubo_penalty(G, K)
    penalty_rqubo = tight_rqubo_penalty(G, K)
    
    # Create models
    print("\nCreating optimization models...")
    model_qubo = docplex_QUBO(G, K, penalty_qubo, name="QUBO_MaxKCut")
    model_rqubo = docplex_RQUBO(G, K, penalty_rqubo, name="RQUBO_MaxKCut")
    
    # Set up QAOA
    print("\nSetting up QAOA...")
    backend = AerSimulator()
    sampler = Sampler()
    sampler.set_options(shots=1024, backend=backend)
    optimizer = COBYLA()
    reps = 2
    initial_point = np.random.rand(2 * reps) * np.pi / 2
    
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps,
        initial_point=initial_point,
    )
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    
    # Run QAOA
    print("\nRunning QAOA optimization...")
    results = run_qaoa_extract_samples(
        [model_qubo, model_rqubo],
        ["QUBO (Tight)", "RQUBO (Tight)"],
        optimizer=qaoa_optimizer
    )
    
    # Analyze results
    print("\n" + "="*50)
    print("Results Analysis")
    print("="*50)
    
    for label, result in results.items():
        samples = result["samples"]
        
        # Filter feasible solutions
        feasible_samples, feas_prob = feasibility_filter(G, K, samples, label)
        
        # Compute expected approximation ratio
        exp_val = expected_value(feasible_samples)
        
        print(f"\n{label}:")
        print(f"  Feasibility probability: {feas_prob:.4f}")
        print(f"  Expected approximation ratio: {exp_val:.4f}")
        print(f"  Number of feasible samples: {len(feasible_samples)}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
