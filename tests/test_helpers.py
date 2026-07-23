"""Sanity tests for hamming_distance_to_feasibility.

Runnable directly (python tests/test_helpers.py) or via pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np
from qiskit_optimization.algorithms import OptimizationResultStatus, SolutionSample

from max_k_cut import feasibility_filter, hamming_distance_to_feasibility


def make_samples(bitstrings, probabilities):
    return [
        SolutionSample(
            x=np.array(x),
            fval=0.0,
            probability=p,
            status=OptimizationResultStatus.SUCCESS,
        )
        for x, p in zip(bitstrings, probabilities)
    ]


def two_node_graph():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    return G


def test_qubo_distances():
    G = two_node_graph()
    K = 3
    samples = make_samples(
        [
            [1, 0, 0, 0, 1, 0],  # both rows one-hot -> d = 0
            [0, 0, 0, 1, 1, 0],  # row sums (0, 2)   -> d = 1 + 1 = 2
            [1, 1, 1, 0, 0, 0],  # row sums (3, 0)   -> d = 2 + 1 = 3
        ],
        [0.5, 0.3, 0.2],
    )
    result = hamming_distance_to_feasibility(G, K, samples, "QUBO (Interpolated)")

    assert result["distances"].tolist() == [0, 2, 3]
    assert np.isclose(result["pmf"].sum(), 1)
    assert result["pmf"].shape == (G.number_of_nodes() * (K - 1) + 1,)
    assert np.isclose(result["expected_distance"], 0.3 * 2 + 0.2 * 3)
    assert np.isclose(result["expected_distance_infeasible"], (0.3 * 2 + 0.2 * 3) / 0.5)

    _, feas_prob = feasibility_filter(G, K, samples, "QUBO (Interpolated)")
    assert np.isclose(result["feasibility_probability"], feas_prob)
    assert np.isclose(result["pmf"][0], 0.5)


def test_rqubo_distances():
    G = two_node_graph()
    K = 3
    samples = make_samples(
        [
            [0, 0, 1, 0],  # row sums (0, 1) -> d = 0 (both feasible in reduced encoding)
            [1, 1, 0, 0],  # row sums (2, 0) -> d = 1
            [1, 1, 1, 1],  # row sums (2, 2) -> d = 2
        ],
        [0.6, 0.3, 0.1],
    )
    result = hamming_distance_to_feasibility(G, K, samples, "RQUBO (Interpolated)")

    assert result["distances"].tolist() == [0, 1, 2]
    assert np.isclose(result["pmf"].sum(), 1)
    assert np.isclose(result["expected_distance"], 0.3 * 1 + 0.1 * 2)
    assert np.isclose(result["expected_distance_infeasible"], (0.3 + 0.2) / 0.4)

    _, feas_prob = feasibility_filter(G, K, samples, "RQUBO (Interpolated)")
    assert np.isclose(result["feasibility_probability"], feas_prob)


def test_all_feasible_gives_nan_conditional():
    G = two_node_graph()
    K = 3
    samples = make_samples([[1, 0, 0, 0, 1, 0]], [1.0])
    result = hamming_distance_to_feasibility(G, K, samples, "QUBO")
    assert result["feasibility_probability"] == 1.0
    assert np.isnan(result["expected_distance_infeasible"])


def test_invalid_label_raises():
    G = two_node_graph()
    samples = make_samples([[1, 0, 0, 0, 1, 0]], [1.0])
    try:
        hamming_distance_to_feasibility(G, 3, samples, "BQO")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for label 'BQO'")


if __name__ == "__main__":
    test_qubo_distances()
    test_rqubo_distances()
    test_all_feasible_gives_nan_conditional()
    test_invalid_label_raises()
    print("All tests passed.")
