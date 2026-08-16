"""Sanity tests for the resource-estimation helpers (max_k_cut/resources.py).

Closed forms for a graph with n nodes, |E| edges, K partitions:
  QUBO cost ZZ terms   = K*|E| + C(K,2) * (#nodes with nonzero penalty)
  RQUBO cost ZZ terms  = (K-1)^2*|E| + C(K-1,2) * (#nodes with nonzero penalty)
  no-constraints ZZ    = K*|E|
  ring XY mixer 2q     = 2*K*n
Isolated nodes have tight penalty 0, so their within-node penalty terms
vanish at t=0 (and reappear at t>0).

Runnable directly (python tests/test_resources.py) or via pytest.
"""

import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np

from max_k_cut import (
    build_model_circuits,
    decompose_for_counting,
    interaction_counts,
    logical_metrics,
    tight_qubo_penalty,
    tight_rqubo_penalty,
)

K = 3


def experiment_graph_0():
    """Graph 0 of the seed-42 experiment set (node 4 is isolated)."""
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edges_from(
        [
            (0, 2, {"weight": 1}),
            (0, 3, {"weight": 1}),
            (1, 3, {"weight": 1}),
            (2, 3, {"weight": 10}),
        ]
    )
    return G


def test_qubit_counts():
    G = experiment_graph_0()
    n = G.number_of_nodes()
    assert build_model_circuits("QUBO", G, K)["num_qubits"] == n * K
    assert build_model_circuits("RQUBO", G, K)["num_qubits"] == n * (K - 1)
    assert build_model_circuits("Dicke RQUBO", G, K)["num_qubits"] == n * (K - 1)
    assert build_model_circuits("XY Mixer", G, K)["num_qubits"] == n * K


def test_logical_cost_counts_closed_form():
    G = experiment_graph_0()
    n, E = G.number_of_nodes(), G.number_of_edges()

    active_qubo = int(np.count_nonzero(tight_qubo_penalty(G, K)))
    qubo = build_model_circuits("QUBO", G, K, t=0.0)
    assert (
        logical_metrics(qubo["cost"])["twoq_count"]
        == K * E + comb(K, 2) * active_qubo
    )

    active_rqubo = int(np.count_nonzero(tight_rqubo_penalty(G, K)))
    rqubo = build_model_circuits("RQUBO", G, K, t=0.0)
    assert (
        logical_metrics(rqubo["cost"])["twoq_count"]
        == (K - 1) ** 2 * E + comb(K - 1, 2) * active_rqubo
    )

    # Dicke RQUBO runs the same cost layer as RQUBO; only the init differs.
    dicke_rqubo = build_model_circuits("Dicke RQUBO", G, K, t=0.0)
    assert (
        logical_metrics(dicke_rqubo["cost"])["twoq_count"]
        == logical_metrics(rqubo["cost"])["twoq_count"]
    )

    xy = build_model_circuits("XY Mixer", G, K)
    assert logical_metrics(xy["cost"])["twoq_count"] == K * E


def test_mixer_and_init_counts():
    G = experiment_graph_0()
    n = G.number_of_nodes()
    xy = build_model_circuits("XY Mixer", G, K)
    assert logical_metrics(xy["mixer"])["twoq_count"] == 2 * K * n
    # Dicke prep decomposes to a fixed number of CX per node (4 for K=3).
    assert logical_metrics(xy["init"])["twoq_count"] == 4 * n
    # Reduced-Dicke prep is a gate-based cascade: (K-2) cry + (K-2) cx per node.
    dicke_rqubo = build_model_circuits("Dicke RQUBO", G, K)
    assert logical_metrics(dicke_rqubo["init"])["twoq_count"] == 2 * (K - 2) * n
    assert logical_metrics(dicke_rqubo["mixer"])["twoq_count"] == 0
    # QUBO/RQUBO have no 2-qubit gates outside the cost layer.
    for name in ("QUBO", "RQUBO"):
        c = build_model_circuits(name, G, K)
        assert logical_metrics(c["init"])["twoq_count"] == 0
        assert logical_metrics(c["mixer"])["twoq_count"] == 0


def test_isolated_node_penalty_terms_at_naive():
    """At t=1 (naive penalty) every node contributes penalty terms."""
    G = experiment_graph_0()
    n, E = G.number_of_nodes(), G.number_of_edges()
    qubo = build_model_circuits("QUBO", G, K, t=1.0)
    assert logical_metrics(qubo["cost"])["twoq_count"] == K * E + comb(K, 2) * n


def test_interaction_counts_match_totals():
    G = experiment_graph_0()
    xy = build_model_circuits("XY Mixer", G, K)
    full = decompose_for_counting(xy["full"])
    pairs = interaction_counts(full)
    assert sum(pairs.values()) == logical_metrics(xy["full"])["twoq_count"]
    assert all(q0 < q1 for q0, q1 in pairs)


def test_hardware_at_least_logical():
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    from max_k_cut import hardware_metrics

    G = experiment_graph_0()
    backend = FakeBrisbane()
    for name in ("RQUBO", "XY Mixer"):
        c = build_model_circuits(name, G, K)
        hw = hardware_metrics(c["full"], backend)
        lg = logical_metrics(c["full"])
        assert hw["twoq_count"] >= lg["twoq_count"]
        assert hw["twoq_depth"] >= lg["twoq_depth"]
        # Reproducibility: seeded transpilation gives identical counts.
        hw2 = hardware_metrics(c["full"], backend)
        assert hw2["twoq_count"] == hw["twoq_count"]


if __name__ == "__main__":
    test_qubit_counts()
    test_logical_cost_counts_closed_form()
    test_mixer_and_init_counts()
    test_isolated_node_penalty_terms_at_naive()
    test_interaction_counts_match_totals()
    test_hardware_at_least_logical()
    print("All resource tests passed.")
