"""
Circuit resource estimation for the Max-K-Cut QAOA model variants.

Reconstructs the QAOA ansatz each model variant runs (the experiments build it
implicitly inside MinimumEigenOptimizer(QAOA(...))) so that 2-qubit gate
counts, 2-qubit depth, and qubit-interaction structure can be compared across
models at both the logical level (native RZZ/RXX/RYY, all-to-all connectivity)
and the hardware level (transpiled to a heavy-hex backend).
"""

import pickle
from collections import Counter

import numpy as np
import networkx as nx

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp

from .models import docplex_QUBO, docplex_RQUBO, docplex_QUBO_no_constraints
from .penalties import interpolated_qubo_penalty, interpolated_rqubo_penalty
from .helpers import (
    create_dicke_initial_state,
    create_reduced_dicke_initial_state,
    create_ring_xy_mixer,
)

# The seven model variants compared in the experiments. "XY Mixer (no agg)" runs
# the identical circuit to "XY Mixer" (only the CVaR aggregation differs), so
# resource metrics are shared between the two.
RESOURCE_MODELS = [
    "QUBO",
    "RQUBO",
    "Dicke QUBO",
    "Dicke RQUBO",
    "Penalty+Mixer QUBO",
    "XY Mixer",
    "XY Mixer (no agg)",
]

# Basis used to decompose qc.initialize (Dicke prep) while keeping the native
# QAOA two-qubit rotations intact for logical-level counting.
_LOGICAL_BASIS = ["rz", "ry", "sx", "x", "h", "cx", "rzz", "rxx", "ryy", "reset"]


def build_cost_operator(dp_model):
    """
    Convert a docplex model to the Ising cost operator QAOA actually runs on,
    following the same path as MinimumEigenOptimizer:
    from_docplex_mp -> QuadraticProgramToQubo -> to_ising.

    Returns:
        (SparsePauliOp, float): the cost operator and constant offset.
    """
    qp = from_docplex_mp(dp_model)
    qubo = QuadraticProgramToQubo().convert(qp)
    op, offset = qubo.to_ising()
    return op, offset


def build_cost_layer(op, gamma):
    """
    Build one QAOA cost layer exp(-i*gamma*H_C) as an explicit circuit:
    rzz(2*coeff*gamma) per ZZ Pauli term and rz(2*coeff*gamma) per Z term.

    Args:
        op (SparsePauliOp): cost operator from build_cost_operator.
        gamma (Parameter): QAOA cost parameter.

    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(op.num_qubits)
    for pauli, coeff in zip(op.paulis, op.coeffs):
        zs = np.where(pauli.z)[0]
        if len(zs) == 2:
            qc.rzz(2 * coeff.real * gamma, int(zs[0]), int(zs[1]))
        elif len(zs) == 1:
            qc.rz(2 * coeff.real * gamma, int(zs[0]))
    return qc


def build_model_circuits(name, G, K, t=0.0):
    """
    Reconstruct the reps=1 QAOA ansatz for one of the seven model variants.

    Args:
        name (str): one of RESOURCE_MODELS.
        G (networkx.Graph): problem graph.
        K (int): number of partitions.
        t (float): penalty interpolation parameter (structure is t-independent;
            only coefficients change).

    Returns:
        dict with keys:
            "num_qubits" (int)
            "init" (QuantumCircuit): initial state (Dicke or H layer)
            "cost" (QuantumCircuit): one cost layer
            "mixer" (QuantumCircuit): one mixer layer (ring XY or rx layer)
            "full" (QuantumCircuit): init | cost | mixer separated by barriers
                labeled "seg" (for segmented counting after transpilation)
            "cost_operator" (SparsePauliOp)
    """
    n = G.number_of_nodes()
    if name in ("QUBO", "Dicke QUBO", "Penalty+Mixer QUBO"):
        dp_model = docplex_QUBO(G, K, interpolated_qubo_penalty(G, K, t), name)
    elif name in ("RQUBO", "Dicke RQUBO"):
        dp_model = docplex_RQUBO(G, K, interpolated_rqubo_penalty(G, K, t), name)
    elif name in ("XY Mixer", "XY Mixer (no agg)"):
        dp_model = docplex_QUBO_no_constraints(G, K, name)
    else:
        raise ValueError(f"Unknown model name: {name}")

    op, _ = build_cost_operator(dp_model)
    num_qubits = op.num_qubits

    gamma = Parameter("gamma")
    beta = Parameter("beta")
    cost = build_cost_layer(op, gamma)

    if name in ("Dicke QUBO", "Penalty+Mixer QUBO", "XY Mixer", "XY Mixer (no agg)"):
        init = create_dicke_initial_state(n, K)
    elif name == "Dicke RQUBO":
        # Gate-based (ry/cry/cx) prep of the 0-hot/1-hot uniform superposition;
        # cry counts as one 2-qubit gate at the logical level, like rzz/rxx/ryy.
        init = create_reduced_dicke_initial_state(n, K)
    else:
        init = QuantumCircuit(num_qubits)
        init.h(range(num_qubits))

    if name in ("Penalty+Mixer QUBO", "XY Mixer", "XY Mixer (no agg)"):
        # Strip the per-node barriers the mixer builder inserts; they would
        # artificially serialize depth and block transpiler optimization.
        mixer = RemoveBarriers()(create_ring_xy_mixer(n, K, beta))
    else:
        # Default QAOA X mixer: rx(2*beta) on every qubit.
        mixer = QuantumCircuit(num_qubits)
        mixer.rx(2 * beta, range(num_qubits))

    full = QuantumCircuit(num_qubits)
    full.compose(init, inplace=True)
    full.barrier(label="seg")
    full.compose(cost, inplace=True)
    full.barrier(label="seg")
    full.compose(mixer, inplace=True)

    return {
        "num_qubits": num_qubits,
        "init": init,
        "cost": cost,
        "mixer": mixer,
        "full": full,
        "cost_operator": op,
    }


def decompose_for_counting(circ):
    """
    Prepare a circuit for logical-level counting: strip barriers and, if the
    circuit contains a non-unitary `initialize` (Dicke prep), decompose it to
    1- and 2-qubit gates while keeping rzz/rxx/ryy native.
    """
    c = RemoveBarriers()(circ)
    if any(inst.operation.name in ("initialize", "state_preparation") for inst in c.data):
        c = transpile(c, basis_gates=_LOGICAL_BASIS, optimization_level=3)
    return c


def logical_metrics(circ):
    """
    Logical-level (all-to-all connectivity) 2-qubit metrics of a circuit.

    Returns:
        dict: {"twoq_count", "twoq_depth"}
    """
    c = decompose_for_counting(circ)
    return {
        "twoq_count": c.num_nonlocal_gates(),
        "twoq_depth": c.depth(lambda inst: inst.operation.num_qubits == 2),
    }


def hardware_metrics(full, backend, seed=42):
    """
    Transpile the full ansatz (parameters unbound, so no angle-specific
    optimizations) to `backend` and report 2-qubit metrics, segmented on the
    "seg"-labeled barriers into init / cost / mixer contributions.

    Per-segment hardware *depth* is not well-defined after routing (segments
    share one layout), so depth is reported for the total circuit only.

    Returns:
        dict: {"init_twoq", "cost_twoq", "mixer_twoq", "twoq_count",
               "twoq_depth", "circuit"}
    """
    tqc = transpile(full, backend=backend, optimization_level=3, seed_transpiler=seed)

    segments = [[]]
    for inst in tqc.data:
        if inst.operation.name == "barrier" and inst.operation.label == "seg":
            segments.append([])
        else:
            segments[-1].append(inst)
    assert len(segments) == 3, (
        f"Expected 3 segments split on 'seg' barriers, got {len(segments)}"
    )

    seg_counts = [
        sum(1 for inst in seg if inst.operation.num_qubits == 2) for seg in segments
    ]
    return {
        "init_twoq": seg_counts[0],
        "cost_twoq": seg_counts[1],
        "mixer_twoq": seg_counts[2],
        "twoq_count": sum(seg_counts),
        "twoq_depth": tqc.depth(lambda inst: inst.operation.num_qubits == 2),
        "circuit": tqc,
    }


def interaction_counts(circ):
    """
    Count 2-qubit gates per qubit pair. Pass a circuit already processed by
    decompose_for_counting (logical) or a transpiled circuit (hardware).

    Returns:
        dict: {(q_lo, q_hi): count}
    """
    counts = Counter()
    for inst in circ.data:
        if inst.operation.num_qubits == 2:
            pair = tuple(sorted(circ.find_bit(q).index for q in inst.qubits))
            counts[pair] += 1
    return dict(counts)


def rebuild_experiment_graphs(pkl_path, num_nodes=5):
    """
    Rebuild the exact experiment graphs from a raw_samples pickle, which
    stores list(G.edges(data=True)) under keys (graph_index, "graph_edges", None).
    Nodes are added explicitly so isolated nodes survive the round trip.

    Returns:
        list[networkx.Graph]
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    graph_ids = sorted(
        k[0] for k in raw
        if isinstance(k, tuple) and len(k) == 3 and k[1] == "graph_edges"
    )
    graphs = []
    for g in graph_ids:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(raw[(g, "graph_edges", None)])
        graphs.append(G)
    return graphs
