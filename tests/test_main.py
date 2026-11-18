"""Tests for QuantumCompiler-Lab: IR, simulator, passes, decomposition,
routing, equivalence, classifier, gates, and seats."""
import cmath
import math

import pytest

import main as m


# ---- IR + simulator
def test_make_gate_normalizes_alias():
    assert m.make_gate("CCX", 0, 1, 2)["name"] == "TOFFOLI"


def test_make_gate_rejects_unknown():
    with pytest.raises(ValueError):
        m.make_gate("MAGIC", 0)


def test_make_gate_rejects_rotation_without_angle():
    with pytest.raises(ValueError):
        m.make_gate("RZ", 0)


def test_bell_state():
    state = m.simulate([m.make_gate("H", 0), m.make_gate("CNOT", 0, 1)], 2, 0)
    assert abs(state[0] - 1 / math.sqrt(2)) < 1e-12
    assert abs(state[3] - 1 / math.sqrt(2)) < 1e-12
    assert m.probabilities(state) == pytest.approx([0.5, 0.0, 0.0, 0.5])


def test_all_single_matrices_unitary():
    for name in m.SINGLE_QUBIT_GATES:
        gate = (m.make_gate(name, 0, angle=0.37) if name in m.ROTATION_GATES
                else m.make_gate(name, 0))
        assert m.is_unitary(m.single_qubit_matrix(gate))


def test_multi_matrices_unitary():
    assert m.is_unitary(m.local_gate_matrix(m.make_gate("CNOT", 0, 1)))
    assert m.is_unitary(m.local_gate_matrix(m.make_gate("CZ", 0, 1)))
    assert m.is_unitary(m.local_gate_matrix(m.make_gate("SWAP", 0, 1)))
    assert m.is_unitary(m.local_gate_matrix(m.make_gate("TOFFOLI", 0, 1, 2)))


def test_circuit_unitary_unitary():
    circ = [m.make_gate("H", 0), m.make_gate("CNOT", 0, 1),
            m.make_gate("RZ", 1, angle=0.4)]
    assert m.is_unitary(m.circuit_unitary(circ, 2))


def test_simulator_qubit_limit():
    with pytest.raises(ValueError):
        m.simulate([m.make_gate("H", 6)], 7, 0)


def test_mat_mul_identity():
    a = m.single_qubit_matrix(m.make_gate("H", 0))
    assert m.mat_max_deviation(m.mat_mul(a, m.identity_matrix(2)), a) == 0.0


# ---- metrics
def test_demo_original_metrics():
    metrics = m.circuit_metrics(m.demo_circuit())
    assert metrics["gate_count"] == 12
    assert metrics["depth"] == 7
    assert metrics["two_qubit_count"] == 3


def test_empty_metrics():
    metrics = m.circuit_metrics([])
    assert metrics["gate_count"] == 0 and metrics["depth"] == 0
    assert metrics["two_qubit_ratio"] == 0.0


def test_depth_uses_parallel_qubits():
    circ = [m.make_gate("H", 0), m.make_gate("H", 1), m.make_gate("H", 0)]
    assert m.circuit_metrics(circ)["depth"] == 2


# ---- passes
def test_h_pair_cancels():
    assert m.cancel_adjacent([m.make_gate("H", 0), m.make_gate("H", 0)]) == []


def test_cancellation_exposes_pair():
    circ = [m.make_gate("X", 0), m.make_gate("H", 1), m.make_gate("H", 1),
            m.make_gate("X", 0)]
    assert m.cancel_adjacent(circ) == []


def test_cnot_pair_cancels():
    circ = [m.make_gate("CNOT", 0, 1), m.make_gate("CNOT", 0, 1)]
    assert m.cancel_adjacent(circ) == []


def test_different_gates_do_not_cancel():
    circ = [m.make_gate("H", 0), m.make_gate("H", 1)]
    assert len(m.cancel_adjacent(circ)) == 2


def test_rotation_merge():
    merged = m.merge_rotations([m.make_gate("RZ", 0, angle=0.2),
                                m.make_gate("RZ", 0, angle=0.7)])
    assert len(merged) == 1 and merged[0]["angle"] == pytest.approx(0.9)


def test_opposite_rotations_cancel():
    merged = m.merge_rotations([m.make_gate("RY", 1, angle=0.4),
                                m.make_gate("RY", 1, angle=-0.4)])
    assert merged == []


def test_rotations_on_different_qubits_do_not_merge():
    circ = [m.make_gate("RZ", 0, angle=0.2), m.make_gate("RZ", 1, angle=0.7)]
    assert len(m.merge_rotations(circ)) == 2


def test_commutation_table():
    assert m.commutes(m.make_gate("Z", 0), m.make_gate("CNOT", 0, 1))
    assert m.commutes(m.make_gate("X", 1), m.make_gate("CNOT", 0, 1))
    assert not m.commutes(m.make_gate("X", 0), m.make_gate("CNOT", 0, 1))
    assert not m.commutes(m.make_gate("Z", 1), m.make_gate("CNOT", 0, 1))
    assert m.commutes(m.make_gate("H", 0), m.make_gate("H", 2))


def test_commutation_reorder_exposes_cancellation():
    circ = [m.make_gate("Z", 0), m.make_gate("CNOT", 0, 1), m.make_gate("Z", 0)]
    reordered = m.commutation_reorder(circ)
    assert [g["name"] for g in reordered] == ["Z", "Z", "CNOT"]
    assert [g["name"] for g in m.cancel_adjacent(reordered)] == ["CNOT"]


def test_optimize_demo():
    result = m.optimize_circuit(m.demo_circuit())
    assert result["after"]["gate_count"] == 3
    assert result["after"]["depth"] == 2
    assert result["after"]["two_qubit_count"] == 1
    assert [g["name"] for g in result["circuit"]] == ["CNOT", "RZ", "H"]


def test_optimize_is_idempotent():
    once = m.optimize_circuit(m.demo_circuit())["circuit"]
    twice = m.optimize_circuit(once)["circuit"]
    assert once == twice


def test_optimized_demo_equivalent_on_all_basis_inputs():
    demo = m.demo_circuit()
    optimized = m.optimize_circuit(demo)["circuit"]
    report = m.equivalence_report(demo, optimized, 4)
    assert report["basis_inputs_tested"] == 16
    assert report["max_deviation"] < 1e-9
    assert report["equivalent"]


# ---- Euler decomposition
def test_euler_sample_reconstruction():
    sample = m.mat_scale(
        m.circuit_unitary([m.make_gate("RZ", 0, angle=1.05),
                           m.make_gate("RY", 0, angle=-0.70),
                           m.make_gate("RZ", 0, angle=0.30)], 1),
        cmath.exp(0.20j))
    result = m.euler_decompose(sample)
    assert result["reconstruction_max_deviation"] < 1e-10


@pytest.mark.parametrize("matrix", [
    m.identity_matrix(2),
    m.single_qubit_matrix(m.make_gate("H", 0)),
    m.single_qubit_matrix(m.make_gate("X", 0)),
    m.single_qubit_matrix(m.make_gate("T", 0)),
    m.single_qubit_matrix(m.make_gate("RY", 0, angle=1.1)),
])
def test_euler_edge_cases(matrix):
    assert m.euler_decompose(matrix)["reconstruction_max_deviation"] < 1e-10


def test_euler_circuit_matches_up_to_global_phase():
    sample = m.circuit_unitary([m.make_gate("RZ", 0, angle=0.9),
                                m.make_gate("RY", 0, angle=0.4)], 1)
    result = m.euler_decompose(sample)
    rebuilt = m.circuit_unitary(result["circuit"], 1)
    # Global phase is physical only up to a scalar; compare projectors
    # via the deviation after aligning the [0][0] phase.
    ratio = sample[0][0] / rebuilt[0][0]
    aligned = m.mat_scale(rebuilt, ratio)
    assert m.mat_max_deviation(sample, aligned) < 1e-9


def test_euler_rejects_non_unitary():
    with pytest.raises(ValueError):
        m.euler_decompose([[1, 1], [0, 1]])


# ---- Toffoli decomposition
def test_toffoli_decomposition_six_cnots():
    decomposed = m.decompose_circuit([m.make_gate("TOFFOLI", 0, 1, 2)])
    assert sum(g["name"] == "CNOT" for g in decomposed) == 6


def test_toffoli_decomposition_equivalent():
    toffoli = [m.make_gate("TOFFOLI", 0, 1, 2)]
    report = m.equivalence_report(toffoli, m.decompose_circuit(toffoli), 3)
    assert report["max_deviation"] < 1e-9 and report["equivalent"]


def test_toffoli_truth_table():
    toffoli = [m.make_gate("TOFFOLI", 0, 1, 2)]
    for basis, expected in [(0, 0), (3, 7), (7, 3), (6, 6), (5, 5)]:
        state = m.simulate(toffoli, 3, basis)
        assert abs(state[expected] - 1) < 1e-12


def test_decompose_circuit_passes_other_gates_through():
    circ = [m.make_gate("H", 0), m.make_gate("TOFFOLI", 0, 1, 2)]
    out = m.decompose_circuit(circ)
    assert out[0]["name"] == "H"
    assert all(g["name"] != "TOFFOLI" for g in out)


# ---- routing
def test_line_graph_shape():
    graph = m.coupling_graph(4, "line")
    assert graph[0] == {1} and graph[3] == {2} and graph[1] == {0, 2}


def test_ring_graph_shape():
    graph = m.coupling_graph(4, "ring")
    assert graph[0] == {1, 3} and graph[3] == {0, 2}


def test_line_routing_swap_count():
    circ = [m.make_gate("H", 0), m.make_gate("CNOT", 0, 3)]
    assert m.route_circuit(circ, 4, "line")["swap_count"] == 2


def test_ring_routing_no_swaps():
    circ = [m.make_gate("H", 0), m.make_gate("CNOT", 0, 3)]
    assert m.route_circuit(circ, 4, "ring")["swap_count"] == 0


def test_routing_equivalence_after_mapping():
    circ = [m.make_gate("H", 0), m.make_gate("CNOT", 0, 3),
            m.make_gate("RZ", 3, angle=0.4), m.make_gate("CNOT", 3, 0)]
    for topology in ("line", "ring"):
        report = m.routed_equivalence_report(circ, topology)
        assert report["equivalent_after_mapping"]
        assert report["max_deviation"] < 1e-9


def test_routing_rejects_toffoli():
    with pytest.raises(ValueError):
        m.route_circuit([m.make_gate("TOFFOLI", 0, 1, 2)], 3, "line")


def test_routing_adjacent_needs_no_swap():
    circ = [m.make_gate("CNOT", 1, 2)]
    result = m.route_circuit(circ, 4, "line")
    assert result["swap_count"] == 0
    assert result["final_mapping"] == {0: 0, 1: 1, 2: 2, 3: 3}


# ---- classifier
def test_classifier_scenarios():
    report = m.classifier_report()
    assert report["small_native_circuit"] == "cheap"
    assert report["line_routed_long_cnot"] == "moderate"
    assert report["swap_storm"] == "expensive"
    assert report["deep_two_qubit_heavy"] == "expensive"


def test_classifier_matches_model():
    for features in [(2, 0.18, 0), (3, 0.18, 0), (8, 0.25, 0),
                     (6, 0.35, 2), (16, 0.6, 0), (10, 0.35, 6)]:
        assert m.classify_circuit(*features) == m.model_label(*features)


def test_classifier_from_metrics():
    metrics = m.optimize_circuit(m.demo_circuit())["after"]
    assert m.classify_metrics(metrics) == "moderate"


# ---- gates + seats
def test_hold_gates_count_and_style():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


@pytest.mark.parametrize("action", [
    "real_hardware_compilation_claim", "production_transpiler_claim",
    "quantum_advantage_claim", "cloud_hardware_submission",
    "fidelity_guarantee_claim",
])
def test_gated_actions_hold(action):
    assert m.gate(action) == "HOLD"


def test_sandbox_proceeds():
    assert m.gate("run_teaching_simulation") == "PROCEED"


def test_argon4_seat():
    decider = m.brains_report()["decider"]
    assert decider["seat"] == "Gemini Argon 4 Extended Thinking"
    assert decider["seat_id"] == "gemini-argon-4"
    assert decider["policy"] == "tighten-only"
    assert decider["can_override_hold"] is False
    assert decider["live_calls"] == 0


def test_claude_and_flow_seats():
    brains = m.brains_report()
    assert brains["advisor"]["can_override_hold"] is False
    assert brains["flow"]["rendered"] is False
    assert brains["flow"]["live"] == 0


def test_run_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks)
    assert len(checks) == 55
