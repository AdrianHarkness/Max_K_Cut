"""Sanity tests for the reduced-Dicke (0-hot / 1-hot) RQUBO initial state.

Runnable directly (python tests/test_initial_states.py) or via pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_optimization.translators import from_docplex_mp

from max_k_cut import (
    create_reduced_dicke_initial_state,
    docplex_RQUBO,
    feasibility_mask_from_qp,
    naive_rqubo_penalty,
    prepare_reduced_dicke_state_qiskit,
)


def test_single_node_amplitudes():
    """Per-node circuit: amplitude 1/sqrt(K) on the 0-hot and each 1-hot state, 0 elsewhere."""
    for K in range(2, 7):
        n = K - 1
        sv = Statevector.from_instruction(prepare_reduced_dicke_state_qiskit(K)).data
        assert sv.shape == (2**n,)
        for z in range(2**n):
            weight = bin(z).count("1")
            if weight <= 1:
                assert np.isclose(sv[z], 1 / np.sqrt(K)), (
                    f"K={K}, z={z:0{n}b}: expected 1/sqrt({K}), got {sv[z]}"
                )
            else:
                assert np.isclose(sv[z], 0), f"K={K}, z={z:0{n}b}: expected 0, got {sv[z]}"


def test_gate_based_no_initialize():
    """Circuit is built from plain gates (transpiles/inverts cleanly, unlike initialize)."""
    for K in range(2, 7):
        qc = prepare_reduced_dicke_state_qiskit(K)
        names = {instr.operation.name for instr in qc.data}
        assert names <= {"ry", "cry", "cx"}, f"unexpected ops: {names}"


def test_full_initial_state_uniform_over_feasible():
    """Full circuit is uniform over all K^n feasible colorings and zero on infeasible states."""
    num_nodes, K = 3, 3
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=1)

    init_qc = create_reduced_dicke_initial_state(num_nodes, K)
    assert init_qc.num_qubits == num_nodes * (K - 1)
    probs = np.abs(Statevector.from_instruction(init_qc).data) ** 2

    qp = from_docplex_mp(docplex_RQUBO(G, K, naive_rqubo_penalty(G, K), "test"))
    mask = feasibility_mask_from_qp(qp, "RQUBO", num_nodes, K)

    num_feasible = K**num_nodes
    assert np.isclose(mask.sum(), num_feasible)
    # all probability mass on feasible states, uniformly distributed
    assert np.isclose(probs @ mask, 1)
    assert np.allclose(probs[mask == 1], 1 / num_feasible)
    assert np.allclose(probs[mask == 0], 0)


def test_per_node_color_marginals_uniform():
    """Each node's register puts probability 1/K on the implied color (0-hot row)."""
    num_nodes, K = 2, 4
    n = K - 1
    init_qc = create_reduced_dicke_initial_state(num_nodes, K)
    probs = np.abs(Statevector.from_instruction(init_qc).data) ** 2
    for node in range(num_nodes):
        zero_hot_prob = 0.0
        for z in range(2 ** init_qc.num_qubits):
            row = (z >> (node * n)) & (2**n - 1)
            if row == 0:
                zero_hot_prob += probs[z]
        assert np.isclose(zero_hot_prob, 1 / K)


def test_invalid_K_raises():
    try:
        prepare_reduced_dicke_state_qiskit(1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for K=1")


if __name__ == "__main__":
    test_single_node_amplitudes()
    test_gate_based_no_initialize()
    test_full_initial_state_uniform_over_feasible()
    test_per_node_color_marginals_uniform()
    test_invalid_K_raises()
    print("All tests passed.")
