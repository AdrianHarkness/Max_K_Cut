"""
Max K-Cut QUBO Reformulations Package

This package implements and compares four different optimization formulations
for the Max K-Cut problem: BQO, QUBO, RBQO, and RQUBO.
"""

from .models import (
    docplex_BQO,
    docplex_QUBO,
    docplex_RBQO,
    docplex_RQUBO,
)

from .penalties import (
    tight_qubo_penalty,
    naive_qubo_penalty,
    interpolated_qubo_penalty,
    tight_rqubo_penalty,
    naive_rqubo_penalty,
    interpolated_rqubo_penalty,
)

from .helpers import (
    generate_graph,
    plot_graph,
    expected_value,
    sample_std,
    get_solution_values,
    rbqo_to_bqo,
    rqubo_to_bqo,
    compute_bqo_objective,
    feasibility_filter,
    repenalize,
    prepare_dicke_state_qiskit,
    create_dicke_initial_state,
    create_full_xy_mixer,
    create_ring_xy_mixer,
)

from .qaoa import run_qaoa_extract_samples

__version__ = "0.1.0"
__all__ = [
    # Models
    "docplex_BQO",
    "docplex_QUBO",
    "docplex_RBQO",
    "docplex_RQUBO",
    # Penalties
    "tight_qubo_penalty",
    "naive_qubo_penalty",
    "interpolated_qubo_penalty",
    "tight_rqubo_penalty",
    "naive_rqubo_penalty",
    "interpolated_rqubo_penalty",
    # Helpers
    "generate_graph",
    "plot_graph",
    "expected_value",
    "sample_std",
    "get_solution_values",
    "rbqo_to_bqo",
    "rqubo_to_bqo",
    "compute_bqo_objective",
    "feasibility_filter",
    "repenalize",
    "prepare_dicke_state_qiskit",
    "create_dicke_initial_state",
    "create_full_xy_mixer",
    "create_ring_xy_mixer",
    # QAOA
    "run_qaoa_extract_samples",
]
