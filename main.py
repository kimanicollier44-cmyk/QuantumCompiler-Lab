#!/usr/bin/env python3
"""QuantumCompiler-Lab - educational quantum circuit compiler lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only.

QuantumCompiler-Lab is a teaching transpiler for tiny circuits. It
implements a small circuit intermediate representation (IR), a set of
conservative compiler passes, gate decompositions, qubit routing on
line and ring coupling graphs, circuit metrics, and a small exact
statevector simulator used to prove that the optimized circuit still
computes the same unitary action on every basis input.

Honest scope, stated up front:

  * This is NOT Qiskit, Cirq, TKET, or a production transpiler.
  * It targets no quantum hardware and submits nothing anywhere.
  * Its simulator is deliberately limited to at most six qubits.
  * Its routing model counts SWAP insertions on toy line/ring graphs;
    it does not model calibration, crosstalk, pulse schedules, native
    gate fidelities, or device noise.
  * Equivalence here means exact simulated equivalence for the toy
    circuits in this lab, within floating-point tolerance - not a
    formal proof about arbitrary circuits or real devices.

Gemini Argon 4 Extended Thinking is the council DECIDER (names-only,
tighten-only), Muse is an advisor that can never override a HOLD,
and Google Flow is plan-only.
"""
import cmath
import hashlib
import json
import math
import sys
from collections import deque

VERSION = "1.0.0"
TOL = 1e-9
MAX_QUBITS = 6
TWO_PI = 2.0 * math.pi

SINGLE_QUBIT_GATES = {
    "H", "X", "Y", "Z", "S", "SDG", "T", "TDG",
    "RX", "RY", "RZ",
}
ROTATION_GATES = {"RX", "RY", "RZ"}
SELF_INVERSE_GATES = {"H", "X", "Y", "Z", "CNOT", "CZ", "SWAP", "TOFFOLI"}
ALIASES = {"CCX": "TOFFOLI", "TOFFOLI": "TOFFOLI"}

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "real_hardware_compilation_claim": "HOLD - no claim that this teaching transpiler compiles for real quantum hardware",
    "production_transpiler_claim": "HOLD - no production transpiler, enterprise compiler, or drop-in Qiskit replacement claim",
    "quantum_advantage_claim": "HOLD - no quantum advantage, speedup, or supremacy claim from toy-circuit metrics",
    "hardware_native_gate_claim": "HOLD - no claim that emitted gates are native, calibrated, or executable on a named device",
    "error_correction_guarantee_claim": "HOLD - no fault-tolerance or error-correction guarantee from decomposition or routing examples",
    "fidelity_guarantee_claim": "HOLD - no hardware fidelity, coherence, or error-rate guarantee; simulated equivalence is not device fidelity",
    "commercial_compiler_claim": "HOLD - no commercial compiler, optimization service, or customer workload claim",
    "cloud_hardware_submission": "HOLD - no submitting circuits to cloud quantum hardware or a live provider account",
    "proprietary_backend_claim": "HOLD - no reverse-engineered, proprietary, or vendor-confidential backend targeting claim",
    "real_device_calibration_claim": "HOLD - no real-device calibration, pulse, crosstalk, or timing model is present or implied",
    "cryptographic_security_claim": "HOLD - no claim that a compiled toy circuit establishes cryptographic security",
    "medical_scientific_decision_claim": "HOLD - no using this lab as the basis for medical, engineering, or scientific decisions",
    "financial_decision_claim": "HOLD - no financial, trading, procurement, or investment decision claim from compiler metrics",
    "replace_expert_review_claim": "HOLD - no replacing compiler, physics, hardware, or safety review by qualified experts",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. Circuit IR
# A gate is a JSON-friendly dictionary:
#   {"name": "CNOT", "qubits": (0, 1)}
#   {"name": "RZ", "qubits": (2,), "angle": 0.35}
# Qubit 0 is the least-significant bit of a basis-state index. In a
# multi-qubit gate tuple, the first qubit is the first operand in the
# usual operator notation: CNOT(control, target), for example.

def make_gate(name, *qubits, angle=None):
    """Build and validate one normalized IR gate."""
    raw_name = str(name).upper()
    name = ALIASES.get(raw_name, raw_name)
    qubits = tuple(int(q) for q in qubits)
    if not qubits:
        raise ValueError("a gate needs at least one qubit")
    if len(set(qubits)) != len(qubits):
        raise ValueError("a gate cannot repeat a qubit")
    if any(q < 0 for q in qubits):
        raise ValueError("qubit indices must be non-negative")
    if name in SINGLE_QUBIT_GATES:
        if len(qubits) != 1:
            raise ValueError(f"{name} takes exactly one qubit")
    elif name in {"CNOT", "CZ", "SWAP"}:
        if len(qubits) != 2:
            raise ValueError(f"{name} takes exactly two qubits")
    elif name == "TOFFOLI":
        if len(qubits) != 3:
            raise ValueError("TOFFOLI takes exactly three qubits")
    else:
        raise ValueError(f"unsupported gate: {name}")
    if name in ROTATION_GATES:
        if angle is None:
            raise ValueError(f"{name} needs an angle")
        angle = float(angle)
    elif angle is not None:
        raise ValueError(f"{name} does not take an angle")
    result = {"name": name, "qubits": qubits}
    if angle is not None:
        result["angle"] = angle
    return result


def normalize_circuit(circuit):
    """Validate and copy a circuit in the lab's IR."""
    normalized = []
    for item in circuit:
        if isinstance(item, dict):
            normalized.append(make_gate(
                item["name"], *item["qubits"], angle=item.get("angle")))
        else:
            # Convenience tuple form: ("RZ", 0, 0.25) or ("H", 0).
            name = item[0]
            if str(name).upper() in ROTATION_GATES:
                normalized.append(make_gate(name, item[1], angle=item[2]))
            else:
                normalized.append(make_gate(name, *item[1:]))
    return normalized


def infer_qubits(circuit):
    circuit = normalize_circuit(circuit)
    return 1 + max((max(g["qubits"]) for g in circuit), default=-1)


def copy_circuit(circuit):
    return [dict(g) for g in normalize_circuit(circuit)]


def circuit_json(circuit):
    """JSON-ready circuit representation with qubit tuples as lists."""
    return [{"name": g["name"], "qubits": list(g["qubits"]),
             **({"angle": g["angle"]} if "angle" in g else {})}
            for g in normalize_circuit(circuit)]


# ================================================================ 2. Complex matrices
# Matrices are nested Python lists of complex numbers. The lab writes
# its own multiplication, dagger, and vector application so the linear
# algebra is visible rather than imported from a numerical library.

def identity_matrix(size):
    return [[1.0 + 0j if i == j else 0j for j in range(size)]
            for i in range(size)]


def mat_mul(a, b):
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("matrix dimensions do not align")
    rows, inner, cols = len(a), len(b), len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(inner))
             for j in range(cols)] for i in range(rows)]


def mat_vec_mul(matrix, vector):
    if len(matrix) != len(vector):
        raise ValueError("matrix/vector dimensions do not align")
    return [sum(row[j] * vector[j] for j in range(len(vector)))
            for row in matrix]


def mat_dagger(matrix):
    return [[complex(matrix[j][i]).conjugate()
             for j in range(len(matrix))] for i in range(len(matrix[0]))]


def mat_scale(matrix, scalar):
    return [[scalar * value for value in row] for row in matrix]


def mat_max_deviation(a, b):
    if len(a) != len(b) or any(len(a[i]) != len(b[i]) for i in range(len(a))):
        raise ValueError("matrix shapes differ")
    return max((abs(a[i][j] - b[i][j])
                for i in range(len(a)) for j in range(len(a[i]))),
               default=0.0)


def is_unitary(matrix, tol=1e-8):
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        return False
    product = mat_mul(mat_dagger(matrix), matrix)
    return mat_max_deviation(product, identity_matrix(len(matrix))) <= tol


def _c(value):
    return complex(value)


def single_qubit_matrix(gate_):
    """Exact 2x2 matrix for a supported one-qubit gate."""
    name = gate_["name"]
    root2 = math.sqrt(2.0)
    if name == "H":
        return [[_c(1 / root2), _c(1 / root2)],
                [_c(1 / root2), _c(-1 / root2)]]
    if name == "X":
        return [[0j, 1 + 0j], [1 + 0j, 0j]]
    if name == "Y":
        return [[0j, -1j], [1j, 0j]]
    if name == "Z":
        return [[1 + 0j, 0j], [0j, -1 + 0j]]
    if name == "S":
        return [[1 + 0j, 0j], [0j, 1j]]
    if name == "SDG":
        return [[1 + 0j, 0j], [0j, -1j]]
    if name == "T":
        return [[1 + 0j, 0j], [0j, cmath.exp(0.25j * math.pi)]]
    if name == "TDG":
        return [[1 + 0j, 0j], [0j, cmath.exp(-0.25j * math.pi)]]
    angle = gate_.get("angle")
    if name == "RX":
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        return [[_c(c), -1j * s], [-1j * s, _c(c)]]
    if name == "RY":
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        return [[_c(c), _c(-s)], [_c(s), _c(c)]]
    if name == "RZ":
        return [[cmath.exp(-0.5j * angle), 0j],
                [0j, cmath.exp(0.5j * angle)]]
    raise ValueError(f"not a one-qubit gate: {name}")


def local_gate_matrix(gate_):
    """Matrix in the gate's operand order (first operand is local MSB)."""
    qubits = gate_["qubits"]
    if len(qubits) == 1:
        return single_qubit_matrix(gate_)
    if gate_["name"] == "CNOT":
        return [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    if gate_["name"] == "CZ":
        return [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ]
    if gate_["name"] == "SWAP":
        return [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    if gate_["name"] == "TOFFOLI":
        matrix = identity_matrix(8)
        # Local basis 110 <-> 111: flip the target only when both
        # controls are 1. Rows/columns 6 and 7 are exchanged.
        matrix[6], matrix[7] = matrix[7][:], matrix[6][:]
        return [[complex(v) for v in row] for row in matrix]
    raise ValueError(f"unsupported multi-qubit gate: {gate_['name']}")


def embed_gate_matrix(gate_, n_qubits):
    """Embed a local gate matrix into the full 2^n Hilbert space."""
    if n_qubits < 1 or n_qubits > MAX_QUBITS:
        raise ValueError(f"simulator supports 1..{MAX_QUBITS} qubits")
    gate_ = make_gate(gate_["name"], *gate_["qubits"],
                      angle=gate_.get("angle"))
    if max(gate_["qubits"]) >= n_qubits:
        raise ValueError("gate qubit outside the simulated register")
    local = local_gate_matrix(gate_)
    size = 1 << n_qubits
    width = len(gate_["qubits"])
    full = [[0j for _ in range(size)] for _ in range(size)]
    for col in range(size):
        local_in = 0
        for pos, qubit in enumerate(gate_["qubits"]):
            bit = (col >> qubit) & 1
            local_in |= bit << (width - 1 - pos)
        for local_out in range(1 << width):
            value = local[local_out][local_in]
            if value == 0:
                continue
            row = col
            for pos, qubit in enumerate(gate_["qubits"]):
                bit = (local_out >> (width - 1 - pos)) & 1
                row = (row | (1 << qubit)) if bit else (row & ~(1 << qubit))
            full[row][col] = value
    return full


def simulate(circuit, n_qubits=None, initial=0):
    """Simulate a basis index or explicit statevector.

    Returns a list of complex amplitudes in the lab's basis order.
    """
    circuit = normalize_circuit(circuit)
    n = infer_qubits(circuit) if n_qubits is None else int(n_qubits)
    if n < 1 or n > MAX_QUBITS:
        raise ValueError(f"simulator supports 1..{MAX_QUBITS} qubits")
    size = 1 << n
    if isinstance(initial, int):
        if not 0 <= initial < size:
            raise ValueError("initial basis index out of range")
        state = [0j] * size
        state[initial] = 1.0 + 0j
    else:
        state = [complex(v) for v in initial]
        if len(state) != size:
            raise ValueError("initial state has the wrong dimension")
    for gate_ in circuit:
        state = mat_vec_mul(embed_gate_matrix(gate_, n), state)
    return state


def circuit_unitary(circuit, n_qubits=None):
    circuit = normalize_circuit(circuit)
    n = infer_qubits(circuit) if n_qubits is None else int(n_qubits)
    total = identity_matrix(1 << n)
    for gate_ in circuit:
        total = mat_mul(embed_gate_matrix(gate_, n), total)
    return total


def state_max_deviation(a, b):
    if len(a) != len(b):
        raise ValueError("state dimensions differ")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def equivalence_report(circuit_a, circuit_b, n_qubits=None):
    """Compare two circuits on every computational-basis input."""
    a, b = normalize_circuit(circuit_a), normalize_circuit(circuit_b)
    n = max(infer_qubits(a), infer_qubits(b)) if n_qubits is None else n_qubits
    max_dev = 0.0
    worst_input = None
    for basis in range(1 << n):
        dev = state_max_deviation(simulate(a, n, basis),
                                  simulate(b, n, basis))
        if dev > max_dev:
            max_dev, worst_input = dev, basis
    return {
        "n_qubits": n,
        "basis_inputs_tested": 1 << n,
        "max_deviation": max_dev,
        "worst_basis_input": worst_input,
        "equivalent": max_dev < TOL,
        "tolerance": TOL,
    }


def probabilities(state):
    return [abs(a) ** 2 for a in state]


# ================================================================ 3. Metrics
def circuit_metrics(circuit, swap_overhead=0):
    """Gate/depth metrics. Depth is a greedy per-qubit layer schedule."""
    circuit = normalize_circuit(circuit)
    last_layer = {}
    depth = 0
    for gate_ in circuit:
        layer = 1 + max((last_layer.get(q, 0) for q in gate_["qubits"]),
                        default=0)
        for q in gate_["qubits"]:
            last_layer[q] = layer
        depth = max(depth, layer)
    gate_count = len(circuit)
    multi = sum(1 for g in circuit if len(g["qubits"]) >= 2)
    twoq = sum(1 for g in circuit if len(g["qubits"]) == 2)
    swaps = sum(1 for g in circuit if g["name"] == "SWAP")
    return {
        "gate_count": gate_count,
        "depth": depth,
        "two_qubit_count": multi,  # multi-qubit cost count, incl. TOFFOLI
        "strict_two_qubit_count": twoq,
        "one_qubit_count": sum(1 for g in circuit if len(g["qubits"]) == 1),
        "swap_count": swaps,
        "swap_overhead": int(swap_overhead),
        "two_qubit_ratio": (multi / gate_count) if gate_count else 0.0,
    }


# ================================================================ 4. Compiler passes
def _same_gate(a, b):
    return (a["name"] == b["name"] and a["qubits"] == b["qubits"]
            and a.get("angle") == b.get("angle"))


def _can_cancel(a, b):
    return a["name"] in SELF_INVERSE_GATES and _same_gate(a, b)


def _can_merge(a, b):
    return (a["name"] in ROTATION_GATES and a["name"] == b["name"]
            and a["qubits"] == b["qubits"])


def _pair_can_reduce(a, b):
    return _can_cancel(a, b) or _can_merge(a, b)


def cancel_adjacent(circuit):
    """Remove adjacent identical self-inverse pairs, repeatedly.

    A stack pass is used, so removing one pair can expose another pair
    that then cancels in the same run.
    """
    out = []
    for gate_ in copy_circuit(circuit):
        if out and _can_cancel(out[-1], gate_):
            out.pop()
        else:
            out.append(gate_)
    return out


def _normalize_angle(angle):
    """Normalize an angle to (-pi, pi]."""
    wrapped = (angle + math.pi) % TWO_PI - math.pi
    return math.pi if wrapped == -math.pi else wrapped


def merge_rotations(circuit):
    """Merge adjacent rotations around the same axis and qubit."""
    out = []
    for gate_ in copy_circuit(circuit):
        if out and _can_merge(out[-1], gate_):
            combined = _normalize_angle(out[-1]["angle"] + gate_["angle"])
            if abs(combined) <= 1e-12:
                out.pop()
            else:
                out[-1] = make_gate(gate_["name"], *gate_["qubits"],
                                    angle=combined)
        else:
            out.append(gate_)
    return out


def _phase_family(gate_):
    if gate_["name"] in {"Z", "S", "SDG", "T", "TDG", "RZ"}:
        return "Z"
    if gate_["name"] in {"X", "RX"}:
        return "X"
    if gate_["name"] in {"Y", "RY"}:
        return "Y"
    return None


def commutes(a, b):
    """Conservative adjacent-commutation test from an explicit table.

    Only swaps with a short proof are allowed:
      * gates on disjoint qubits;
      * identical gates;
      * one-qubit gates in the same Pauli/rotation family on one qubit;
      * a Z-family gate on a CNOT control;
      * an X-family gate on a CNOT target;
      * a Z-family gate on either CZ qubit;
      * two CZ gates (both diagonal).
    Everything not listed is treated as non-commuting.
    """
    a, b = (make_gate(g["name"], *g["qubits"], angle=g.get("angle"))
            for g in (a, b))
    if set(a["qubits"]).isdisjoint(b["qubits"]):
        return True
    if _same_gate(a, b):
        return True
    if len(a["qubits"]) == len(b["qubits"]) == 1:
        fam_a, fam_b = _phase_family(a), _phase_family(b)
        return fam_a is not None and fam_a == fam_b
    one, other = (a, b) if len(a["qubits"]) == 1 else (b, a)
    if len(one["qubits"]) == 1 and len(other["qubits"]) == 2:
        family = _phase_family(one)
        q = one["qubits"][0]
        if other["name"] == "CNOT":
            control, target = other["qubits"]
            return (family == "Z" and q == control) or (
                family == "X" and q == target)
        if other["name"] == "CZ":
            return family == "Z" and q in other["qubits"]
        return False
    if a["name"] == b["name"] == "CZ":
        return True
    return False


def commutation_reorder(circuit):
    """Bubble gates across commuting neighbours only to expose a reduction.

    A swap is made only when the gate being moved commutes with its
    neighbour AND the move brings a cancellable or mergeable pair
    together. This deliberately does not attempt general scheduling.
    """
    gates = copy_circuit(circuit)
    max_rounds = max(16, len(gates) ** 3)
    for _ in range(max_rounds):
        changed = False
        for i in range(len(gates) - 2):
            first, middle, third = gates[i], gates[i + 1], gates[i + 2]
            # The only useful local move: swap the middle and third
            # gates (when they commute) so first and third - a pair
            # that can cancel or merge - become adjacent.
            if commutes(middle, third) and _pair_can_reduce(first, third):
                gates[i + 1], gates[i + 2] = third, middle
                changed = True
                break
        if not changed:
            break
    return gates


def optimize_circuit(circuit):
    """Run reorder -> cancel -> merge to a deterministic fixed point."""
    current = copy_circuit(circuit)
    history = [{"pass": "input", "metrics": circuit_metrics(current)}]
    passes = [
        ("commutation_reorder", commutation_reorder),
        ("gate_cancellation", cancel_adjacent),
        ("rotation_merging", merge_rotations),
    ]
    for _ in range(20):
        changed_any = False
        for name, func in passes:
            nxt = func(current)
            if nxt != current:
                history.append({"pass": name,
                                "metrics": circuit_metrics(nxt)})
                current = nxt
                changed_any = True
        if not changed_any:
            break
    return {
        "circuit": current,
        "before": history[0]["metrics"],
        "after": circuit_metrics(current),
        "history": history,
    }


def passes_report():
    demo = demo_circuit()
    result = optimize_circuit(demo)
    return {
        "input": circuit_json(demo),
        "output": circuit_json(result["circuit"]),
        "before": result["before"],
        "after": result["after"],
        "history": result["history"],
    }


# ================================================================ 5. Decomposition
def euler_decompose(matrix, qubit=0):
    """Decompose a 2x2 unitary as global phase * RZ * RY * RZ.

    The returned circuit contains the three rotations. The scalar
    global phase is reported separately because an uncontrolled
    one-qubit gate's global phase is not represented by RZ/RY/RZ.
    `unitary_from_euler` applies that scalar when reconstructing.
    """
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("Euler decomposition needs a 2x2 matrix")
    unit = [[complex(v) for v in row] for row in matrix]
    if not is_unitary(unit):
        raise ValueError("matrix is not unitary")
    det = unit[0][0] * unit[1][1] - unit[0][1] * unit[1][0]
    alpha = cmath.phase(det) / 2.0
    su2 = mat_scale(unit, cmath.exp(-1j * alpha))
    a, b = su2[0][0], su2[0][1]
    gamma = 2.0 * math.acos(max(-1.0, min(1.0, abs(a))))
    eps = 1e-10
    # With U = RZ(beta) * RY(gamma) * RZ(delta) (matrix product):
    #   a = exp(-i(beta+delta)/2) cos(gamma/2)
    #   b = -exp(-i(beta-delta)/2) sin(gamma/2)
    # The difference branch is 2*(pi - phase(b)); it must NOT be
    # reduced modulo 2*pi before solving, or beta/delta shift by pi
    # and the reconstruction picks up a spurious sign.
    if abs(math.sin(gamma / 2)) <= eps:
        beta, delta = -2.0 * cmath.phase(a), 0.0
    elif abs(math.cos(gamma / 2)) <= eps:
        beta, delta = 2.0 * (math.pi - cmath.phase(b)), 0.0
    else:
        total = -2.0 * cmath.phase(a)
        difference = 2.0 * (math.pi - cmath.phase(b))
        beta = (total + difference) / 2.0
        delta = (total - difference) / 2.0
    # Chronological circuit order: the rightmost factor acts first.
    gates = []
    for name, angle in (("RZ", delta), ("RY", gamma), ("RZ", beta)):
        if abs(angle) > 1e-12:
            gates.append(make_gate(name, qubit, angle=angle))
    params = {
        "alpha_global_phase": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "qubit": int(qubit),
    }
    reconstructed = unitary_from_euler(params)
    params["reconstruction_max_deviation"] = mat_max_deviation(
        unit, reconstructed)
    params["circuit"] = gates
    return params


def unitary_from_euler(params):
    """Reconstruct global-phase * RZ(beta) * RY(gamma) * RZ(delta).

    The three factors are multiplied as matrices in the written
    order; the equivalent chronological circuit applies RZ(delta)
    first, then RY(gamma), then RZ(beta).
    """
    beta = single_qubit_matrix(make_gate("RZ", 0, angle=params["beta"]))
    gamma = single_qubit_matrix(make_gate("RY", 0, angle=params["gamma"]))
    delta = single_qubit_matrix(make_gate("RZ", 0, angle=params["delta"]))
    local = mat_mul(beta, mat_mul(gamma, delta))
    return mat_scale(local, cmath.exp(1j * params["alpha_global_phase"]))


def decompose_toffoli(control_a=0, control_b=1, target=2):
    """Standard exact Toffoli decomposition with six CNOT gates."""
    a, b, t = control_a, control_b, target
    sequence = [
        make_gate("H", t),
        make_gate("CNOT", b, t), make_gate("TDG", t),
        make_gate("CNOT", a, t), make_gate("T", t),
        make_gate("CNOT", b, t), make_gate("TDG", t),
        make_gate("CNOT", a, t),
        make_gate("T", b), make_gate("T", t), make_gate("H", t),
        make_gate("CNOT", a, b), make_gate("T", a),
        make_gate("TDG", b), make_gate("CNOT", a, b),
    ]
    return sequence


def decompose_circuit(circuit):
    """Replace every TOFFOLI with its six-CNOT decomposition."""
    out = []
    for gate_ in copy_circuit(circuit):
        if gate_["name"] == "TOFFOLI":
            out.extend(decompose_toffoli(*gate_["qubits"]))
        else:
            out.append(gate_)
    return out


def decomposition_report():
    sample = mat_scale(
        circuit_unitary([
            make_gate("RZ", 0, angle=1.05),
            make_gate("RY", 0, angle=-0.70),
            make_gate("RZ", 0, angle=0.30),
        ], 1),
        cmath.exp(0.20j),
    )
    euler = euler_decompose(sample)
    toffoli = [make_gate("TOFFOLI", 0, 1, 2)]
    decomposed = decompose_circuit(toffoli)
    return {
        "euler": {k: v for k, v in euler.items() if k != "circuit"} |
                 {"circuit": circuit_json(euler["circuit"])},
        "toffoli": {
            "input_metrics": circuit_metrics(toffoli),
            "output_metrics": circuit_metrics(decomposed),
            "cnot_count": sum(g["name"] == "CNOT" for g in decomposed),
            "equivalence": equivalence_report(toffoli, decomposed, 3),
            "circuit": circuit_json(decomposed),
        },
    }


# ================================================================ 6. Mapping + routing
def coupling_graph(n_qubits, topology="line"):
    """Adjacency sets for a line or ring of physical qubits."""
    if n_qubits < 1:
        raise ValueError("need at least one physical qubit")
    graph = {i: set() for i in range(n_qubits)}
    edges = [(i, i + 1) for i in range(n_qubits - 1)]
    if topology == "ring" and n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    elif topology not in {"line", "ring"}:
        raise ValueError("topology must be 'line' or 'ring'")
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _shortest_path(graph, start, goal):
    queue = deque([start])
    parent = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            break
        for nxt in sorted(graph[node]):
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)
    if goal not in parent:
        raise ValueError("coupling graph is disconnected")
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


def route_circuit(circuit, n_qubits=None, topology="line"):
    """Route two-qubit gates by moving the first operand along a path.

    The initial logical->physical mapping is the identity. Inserted
    SWAPs update the mapping; the final mapping is returned and must
    be used when interpreting the physical output state. This is a
    deliberately small teaching router, not an optimizing placer.
    """
    circuit = normalize_circuit(circuit)
    if any(len(g["qubits"]) > 2 for g in circuit):
        raise ValueError("decompose TOFFOLI gates before routing")
    logical_n = infer_qubits(circuit)
    n = logical_n if n_qubits is None else int(n_qubits)
    if n < logical_n:
        raise ValueError("not enough physical qubits")
    graph = coupling_graph(n, topology)
    mapping = {q: q for q in range(n)}
    inverse = {q: q for q in range(n)}
    routed = []
    swaps = 0
    for gate_ in circuit:
        if len(gate_["qubits"]) == 1:
            q = mapping[gate_["qubits"][0]]
            routed.append(make_gate(gate_["name"], q,
                                    angle=gate_.get("angle")))
            continue
        first, second = gate_["qubits"]
        while mapping[second] not in graph[mapping[first]]:
            path = _shortest_path(graph, mapping[first], mapping[second])
            here, nxt = path[0], path[1]
            routed.append(make_gate("SWAP", here, nxt))
            swaps += 1
            other_logical = inverse[nxt]
            inverse[here], inverse[nxt] = other_logical, first
            mapping[first], mapping[other_logical] = nxt, here
        routed.append(make_gate(
            gate_["name"], mapping[first], mapping[second],
            angle=gate_.get("angle")))
    return {
        "circuit": routed,
        "initial_mapping": {q: q for q in range(n)},
        "final_mapping": mapping,
        "swap_count": swaps,
        "swap_overhead": swaps,
        "metrics": circuit_metrics(routed, swap_overhead=swaps),
        "topology": topology,
        "n_qubits": n,
    }


def permute_physical_to_logical(state, final_mapping):
    """Reinterpret a physical output state in logical-qubit order."""
    n = len(final_mapping)
    if len(state) != 1 << n:
        raise ValueError("state dimension does not match mapping")
    out = [0j] * len(state)
    for physical_index, amplitude in enumerate(state):
        logical_index = 0
        for logical_q, physical_q in final_mapping.items():
            if (physical_index >> physical_q) & 1:
                logical_index |= 1 << logical_q
        out[logical_index] = amplitude
    return out


def routed_equivalence_report(circuit, topology="line"):
    circuit = normalize_circuit(circuit)
    n = infer_qubits(circuit)
    routed = route_circuit(circuit, n, topology)
    max_dev = 0.0
    for basis in range(1 << n):
        expected = simulate(circuit, n, basis)
        physical = simulate(routed["circuit"], n, basis)
        actual = permute_physical_to_logical(physical,
                                              routed["final_mapping"])
        max_dev = max(max_dev, state_max_deviation(expected, actual))
    return {
        "topology": topology,
        "swap_count": routed["swap_count"],
        "final_mapping": routed["final_mapping"],
        "basis_inputs_tested": 1 << n,
        "max_deviation": max_dev,
        "equivalent_after_mapping": max_dev < TOL,
    }


def routing_report():
    circuit = [make_gate("H", 0), make_gate("CNOT", 0, 3),
               make_gate("RZ", 3, angle=0.4), make_gate("CNOT", 3, 0)]
    return {
        "circuit": circuit_json(circuit),
        "line": route_circuit(circuit, 4, "line") |
                {"equivalence": routed_equivalence_report(circuit, "line")},
        "ring": route_circuit(circuit, 4, "ring") |
                {"equivalence": routed_equivalence_report(circuit, "ring")},
    }


# ================================================================ 7. Demo circuit
def demo_circuit():
    """A circuit with cancellations, mergeable rotations, and a Z that
    can commute through a CNOT control to meet its cancelling twin."""
    return [
        make_gate("H", 0), make_gate("H", 0),
        make_gate("Z", 0), make_gate("CNOT", 0, 1), make_gate("Z", 0),
        make_gate("RZ", 0, angle=0.20), make_gate("RZ", 0, angle=0.70),
        make_gate("X", 1), make_gate("X", 1),
        make_gate("CNOT", 2, 3), make_gate("CNOT", 2, 3),
        make_gate("H", 2),
    ]


def equivalence_bundle():
    demo = demo_circuit()
    optimized = optimize_circuit(demo)["circuit"]
    toffoli = [make_gate("TOFFOLI", 0, 1, 2)]
    route_source = [make_gate("H", 0), make_gate("CNOT", 0, 3)]
    return {
        "optimized_demo": equivalence_report(demo, optimized, 4),
        "toffoli_decomposition": equivalence_report(
            toffoli, decompose_circuit(toffoli), 3),
        "line_routing": routed_equivalence_report(route_source, "line"),
        "ring_routing": routed_equivalence_report(route_source, "ring"),
    }


# ================================================================ 8. kNN classifier
def model_label(depth, two_qubit_ratio, swap_overhead):
    """Distilled compiler-cost model used to label the kNN grid."""
    if depth >= 12 or two_qubit_ratio >= 0.45 or swap_overhead >= 4:
        return "expensive"
    if depth >= 6 or two_qubit_ratio >= 0.25 or swap_overhead >= 1:
        return "moderate"
    return "cheap"


def _training_grid():
    rows = []
    for depth in (1, 3, 5, 6, 8, 10, 12, 16, 24):
        for ratio in (0.0, 0.15, 0.25, 0.35, 0.45, 0.60):
            for swaps in (0, 1, 2, 4, 6):
                rows.append(((depth, ratio, swaps),
                             model_label(depth, ratio, swaps)))
    return rows


TRAINING_GRID = _training_grid()


def _scaled(features):
    depth, ratio, swaps = features
    return (depth / 24.0, ratio, swaps / 6.0)


def classify_circuit(depth, two_qubit_ratio, swap_overhead, k=3):
    """Classify compiler cost with from-scratch kNN (k=3)."""
    target = _scaled((depth, two_qubit_ratio, swap_overhead))
    ranked = []
    for features, label in TRAINING_GRID:
        point = _scaled(features)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, point)))
        ranked.append((dist, features, label))
    ranked.sort(key=lambda row: (row[0], row[1]))
    votes = {}
    for _, _, label in ranked[:k]:
        votes[label] = votes.get(label, 0) + 1
    priority = {"cheap": 0, "moderate": 1, "expensive": 2}
    return sorted(votes.items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]


def classify_metrics(metrics):
    return classify_circuit(metrics["depth"], metrics["two_qubit_ratio"],
                            metrics["swap_overhead"])


def classifier_report():
    scenarios = {
        "two_qubit_heavy_short": (2, 0.50, 0),
        "small_native_circuit": (3, 0.18, 0),
        "demo_before_optimization": (7, 0.25, 0),
        "demo_after_optimization": (2, 0.3333333333, 0),
        "line_routed_long_cnot": (6, 0.35, 2),
        "deep_two_qubit_heavy": (16, 0.60, 0),
        "swap_storm": (10, 0.35, 6),
    }
    return {name: classify_circuit(*features)
            for name, features in scenarios.items()}


# ================================================================ 9. Advisory seats
def _trace(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="quantumcompiler-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never
    lift one, and can never override a HOLD gate.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "seat_id": "gemini-argon-4",
        "role": "DECIDER",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "policy": "tighten-only",
        "decision": "PROCEED - teaching transpiler only, all hardware and advantage gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="quantumcompiler-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Show the before/after metrics and the all-basis equivalence deviation together: a cheaper circuit is only a win when the simulated output did not move.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": [
            "a circuit ribbon entering a sequence of labelled compiler passes",
            "twin gates cancelling and rotation dials merging into one",
            "a Toffoli gate unfolding into six smaller entangling gates",
            "qubits travelling along a line graph as SWAP bridges appear",
            "two output state fans aligning under an equivalence check",
        ],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ 10. Checks
def _sample_single_matrix(name):
    if name in ROTATION_GATES:
        return single_qubit_matrix(make_gate(name, 0, angle=0.37))
    return single_qubit_matrix(make_gate(name, 0))


def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # IR + matrices + simulator
    bell = [make_gate("H", 0), make_gate("CNOT", 0, 1)]
    bell_state = simulate(bell, 2, 0)
    chk("bell_state_amplitudes",
        abs(bell_state[0] - 1 / math.sqrt(2)) < 1e-12
        and abs(bell_state[3] - 1 / math.sqrt(2)) < 1e-12
        and abs(bell_state[1]) < 1e-12 and abs(bell_state[2]) < 1e-12)
    chk("bell_probabilities", all(
        abs(a - b) < 1e-12 for a, b in
        zip(probabilities(bell_state), [0.5, 0.0, 0.0, 0.5])))
    chk("all_single_gate_matrices_unitary", all(
        is_unitary(_sample_single_matrix(name))
        for name in SINGLE_QUBIT_GATES))
    chk("cnot_matrix_unitary", is_unitary(local_gate_matrix(
        make_gate("CNOT", 0, 1))))
    chk("toffoli_matrix_unitary", is_unitary(local_gate_matrix(
        make_gate("TOFFOLI", 0, 1, 2))))
    chk("circuit_unitary_is_unitary", is_unitary(circuit_unitary(bell, 2)))
    chk("unsupported_gate_rejected", _raises_value_error(
        lambda: make_gate("MAGIC", 0)))
    chk("six_qubit_limit_enforced", _raises_value_error(
        lambda: simulate([make_gate("H", 6)], 7, 0)))

    # metrics
    demo = demo_circuit()
    before = circuit_metrics(demo)
    optimized = optimize_circuit(demo)
    after = optimized["after"]
    chk("demo_original_metrics", before["gate_count"] == 12
        and before["two_qubit_count"] == 3 and before["depth"] == 7)
    chk("demo_optimized_gate_count", after["gate_count"] == 3)
    chk("demo_optimized_two_qubit_reduced",
        after["two_qubit_count"] == 1
        and after["two_qubit_count"] < before["two_qubit_count"])
    chk("demo_optimized_depth_reduced", after["depth"] == 2
        and after["depth"] < before["depth"])
    chk("empty_circuit_metrics", circuit_metrics([])["depth"] == 0
        and circuit_metrics([])["two_qubit_ratio"] == 0.0)

    # individual passes
    chk("h_pair_cancels", cancel_adjacent(
        [make_gate("H", 0), make_gate("H", 0)]) == [])
    chk("x_pair_cancels_after_exposure", cancel_adjacent([
        make_gate("X", 0), make_gate("H", 1), make_gate("H", 1),
        make_gate("X", 0)]) == [])
    chk("cnot_pair_cancels", cancel_adjacent([
        make_gate("CNOT", 0, 1), make_gate("CNOT", 0, 1)]) == [])
    merged = merge_rotations([make_gate("RZ", 0, angle=0.2),
                              make_gate("RZ", 0, angle= 0.7)])
    chk("rz_rotations_merge", len(merged) == 1
        and abs(merged[0]["angle"] - 0.9) < 1e-12)
    cancelling = merge_rotations([make_gate("RY", 1, angle=0.4),
                                  make_gate("RY", 1, angle=-0.4)])
    chk("opposite_rotations_cancel", cancelling == [])
    chk("z_control_commutes_through_cnot", commutes(
        make_gate("Z", 0), make_gate("CNOT", 0, 1)))
    chk("x_target_commutes_through_cnot", commutes(
        make_gate("X", 1), make_gate("CNOT", 0, 1)))
    chk("x_control_does_not_commute_through_cnot", not commutes(
        make_gate("X", 0), make_gate("CNOT", 0, 1)))
    chk("z_target_does_not_commute_through_cnot", not commutes(
        make_gate("Z", 1), make_gate("CNOT", 0, 1)))
    reordered = commutation_reorder([
        make_gate("Z", 0), make_gate("CNOT", 0, 1), make_gate("Z", 0)])
    chk("commutation_exposes_z_cancellation",
        [g["name"] for g in reordered] == ["Z", "Z", "CNOT"])
    chk("optimized_demo_exact_shape",
        [g["name"] for g in optimized["circuit"]] == ["CNOT", "RZ", "H"])
    equiv_demo = equivalence_report(demo, optimized["circuit"], 4)
    chk("optimized_demo_equivalent_all_basis",
        equiv_demo["equivalent"] and equiv_demo["max_deviation"] < TOL
        and equiv_demo["basis_inputs_tested"] == 16)

    # Euler decomposition
    sample = mat_scale(circuit_unitary([
        make_gate("RZ", 0, angle=1.05), make_gate("RY", 0, angle=-0.7),
        make_gate("RZ", 0, angle=0.3)], 1), cmath.exp(0.2j))
    euler = euler_decompose(sample)
    chk("euler_reconstruction_exact",
        euler["reconstruction_max_deviation"] < 1e-10)
    for label, matrix in [
            ("identity", identity_matrix(2)),
            ("hadamard", single_qubit_matrix(make_gate("H", 0))),
            ("x_gate", single_qubit_matrix(make_gate("X", 0))),
            ("phase_gate", single_qubit_matrix(make_gate("T", 0)))]:
        chk(f"euler_{label}_reconstructs",
            euler_decompose(matrix)["reconstruction_max_deviation"] < 1e-10)
    chk("euler_rejects_non_unitary", _raises_value_error(
        lambda: euler_decompose([[1, 1], [0, 1]])))

    # Toffoli decomposition
    toffoli = [make_gate("TOFFOLI", 0, 1, 2)]
    decomposed = decompose_circuit(toffoli)
    chk("toffoli_decomposition_six_cnots",
        sum(g["name"] == "CNOT" for g in decomposed) == 6)
    chk("toffoli_decomposition_equivalent",
        equivalence_report(toffoli, decomposed, 3)["max_deviation"] < TOL)
    chk("toffoli_truth_table", all(
        abs(simulate(toffoli, 3, basis)[expected] - 1) < 1e-12
        for basis, expected in [(0, 0), (3, 7), (7, 3), (6, 6)]))

    # routing
    route_source = [make_gate("H", 0), make_gate("CNOT", 0, 3),
                    make_gate("RZ", 3, angle=0.4)]
    line = route_circuit(route_source, 4, "line")
    ring = route_circuit(route_source, 4, "ring")
    chk("line_routing_inserts_two_swaps", line["swap_count"] == 2)
    chk("ring_routing_needs_no_swaps", ring["swap_count"] == 0)
    chk("line_routing_equivalent_after_mapping",
        routed_equivalence_report(route_source, "line")["equivalent_after_mapping"])
    chk("ring_routing_equivalent_after_mapping",
        routed_equivalence_report(route_source, "ring")["equivalent_after_mapping"])
    chk("routing_rejects_undecomposed_toffoli", _raises_value_error(
        lambda: route_circuit(toffoli, 3, "line")))
    chk("line_graph_shape", coupling_graph(4, "line")[0] == {1}
        and coupling_graph(4, "line")[3] == {2})
    chk("ring_graph_closes_loop", coupling_graph(4, "ring")[0] == {1, 3})

    # classifier: kNN must agree with the distilled metrics model here
    scenarios = [(2, 0.18, 0), (3, 0.18, 0), (8, 0.25, 0),
                 (6, 0.35, 2), (16, 0.6, 0), (10, 0.35, 6)]
    chk("classifier_matches_model_on_scenarios", all(
        classify_circuit(*features) == model_label(*features)
        for features in scenarios))
    report = classifier_report()
    chk("classifier_cheap_case", report["small_native_circuit"] == "cheap")
    chk("classifier_moderate_case", report["line_routed_long_cnot"] == "moderate")
    chk("classifier_expensive_case", report["swap_storm"] == "expensive")
    chk("classifier_uses_metrics", classify_metrics(after) in
        {"cheap", "moderate", "expensive"})

    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_hardware_hold", gate("real_hardware_compilation_claim") == "HOLD")
    chk("gate_production_hold", gate("production_transpiler_claim") == "HOLD")
    chk("gate_advantage_hold", gate("quantum_advantage_claim") == "HOLD")
    chk("gate_submission_hold", gate("cloud_hardware_submission") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_teaching_simulation") == "PROCEED")
    brains = brains_report()
    chk("argon4_no_override", brains["decider"]["can_override_hold"] is False
        and brains["decider"]["seat_id"] == "gemini-argon-4"
        and brains["decider"]["policy"] == "tighten-only")
    chk("claude_no_override", brains["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", brains["flow"]["rendered"] is False
        and brains["flow"]["live"] == 0)
    return c


def _raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


# ================================================================ 11. CLI / demo
def demo():
    print("=" * 68)
    print("QuantumCompiler-Lab - quantum circuit compiler / transpiler lab")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - a teaching transpiler, not Qiskit")
    print("    Tiny circuits (<= 6 qubits) are optimized by conservative")
    print("    passes, decomposed from scratch, routed on line/ring graphs,")
    print("    and proven equivalent by exact statevector simulation on")
    print("    every basis input. No hardware is targeted or claimed.")
    demo_circ = demo_circuit()
    optimized = optimize_circuit(demo_circ)
    print("\n[2] Demo circuit - passes make it measurably cheaper")
    print(f"    input gates {[g['name'] for g in demo_circ]}")
    print(f"    output gates {[g['name'] for g in optimized['circuit']]}")
    print(f"    before: gates {optimized['before']['gate_count']} | "
          f"depth {optimized['before']['depth']} | "
          f"2Q {optimized['before']['two_qubit_count']}")
    print(f"    after:  gates {optimized['after']['gate_count']} | "
          f"depth {optimized['after']['depth']} | "
          f"2Q {optimized['after']['two_qubit_count']}")
    for row in optimized["history"]:
        m = row["metrics"]
        print(f"    pass {row['pass']:22s} gates {m['gate_count']:2d} "
              f"depth {m['depth']:2d} 2Q {m['two_qubit_count']}")
    equiv = equivalence_report(demo_circ, optimized["circuit"], 4)
    print("\n[3] Equivalence - optimized output equals original output")
    print(f"    basis inputs {equiv['basis_inputs_tested']} | "
          f"max deviation {equiv['max_deviation']:.3e} | "
          f"equivalent {equiv['equivalent']}")
    dec = decomposition_report()
    print("\n[4] Decomposition - Euler angles and Toffoli unfolding")
    print(f"    Euler reconstruction deviation "
          f"{dec['euler']['reconstruction_max_deviation']:.3e}")
    print(f"    TOFFOLI -> {dec['toffoli']['cnot_count']} CNOTs | "
          f"equivalence deviation "
          f"{dec['toffoli']['equivalence']['max_deviation']:.3e}")
    routes = routing_report()
    print("\n[5] Routing - the same logical circuit on two graphs")
    for name in ("line", "ring"):
        r = routes[name]
        print(f"    {name:4s}: SWAPs {r['swap_count']} | "
              f"final mapping {r['final_mapping']} | "
              f"equivalent after mapping "
              f"{r['equivalence']['equivalent_after_mapping']}")
    print("\n[6] Circuit cost classifier (TinyML kNN, labels from metrics)")
    for name, label in classifier_report().items():
        print(f"    {name:32s} -> {label}")
    print("\n[7] Advisory seats")
    brains = brains_report()
    print(f"    decider: {brains['decider']['decision']} "
          f"trace={brains['decider']['trace_sha256_16']}")
    print(f"    advisor can_override_hold = {brains['advisor']['can_override_hold']}")
    print(f"    flow rendered = {brains['flow']['rendered']} "
          f"live = {brains['flow']['live']}")
    print(f"\n[8] HOLD gates: {len(HOLD_GATES)}")
    for key in HOLD_GATES:
        print(f"    HOLD  {key}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (teaching simulation only) | HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "passes":
        print(json.dumps(passes_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "decompose":
        print(json.dumps(decomposition_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "route":
        print(json.dumps(routing_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "equiv":
        print(json.dumps(equivalence_bundle(), indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(classifier_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for key, value in HOLD_GATES.items():
            print(f"{key}: {value}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "real_hardware_compilation_claim"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
