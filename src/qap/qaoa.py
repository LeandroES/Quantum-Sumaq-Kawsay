"""QAOA en Qiskit para el QUBO del QAP (instancias o subproblemas de hasta ~16 qubits).

Qiskit construye el Hamiltoniano de Ising (SparsePauliOp), el circuito QAOA
(capas de costo exp(-i*gamma*H) con RZ/RZZ + mezclador RX, ver qaoa_circuit) y lo simula
(Statevector durante la optimización; StatevectorSampler con shots para el muestreo final).
El optimizador de parámetros (COBYLA de SciPy) y la decodificación son clásicos.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector

from .qubo import QUBO


def qaoa_circuit(H: SparsePauliOp, reps: int) -> QuantumCircuit:
    """Circuito QAOA explícito: |+>^n, capas exp(-i γ H) (RZ/RZZ) y mezclador RX(2β).

    Equivale a QAOAAnsatz(H, reps) pero con puertas nativas, lo que hace la simulación
    mucho más rápida (QAOAAnsatz usa PauliEvolutionGate, que Statevector exponencia).
    Parámetros: β[0..p-1], γ[0..p-1] (mismo orden que QAOAAnsatz).
    """
    n = H.num_qubits
    beta = ParameterVector("β", reps)
    gamma = ParameterVector("γ", reps)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    terms = []
    for label, coeff in H.to_list():
        zs = [n - 1 - i for i, ch in enumerate(label) if ch == "Z"]
        if zs:
            terms.append((zs, float(np.real(coeff))))
    for r in range(reps):
        for zs, c in terms:
            if len(zs) == 1:
                qc.rz(2 * gamma[r] * c, zs[0])
            else:
                qc.rzz(2 * gamma[r] * c, zs[0], zs[1])
        qc.rx(2 * beta[r], range(n))
    return qc


@dataclass
class QAOAResult:
    reps: int
    lam: float
    objective: str
    params: list
    n_qubits: int
    n_evals: int
    opt_seconds: float
    shots: int
    feasible_fraction: float
    prob_optimal: float
    best_perm: list | None
    best_cost: float | None
    best_energy_sampled: float
    expected_energy: float
    circuit_depth: int
    two_qubit_gates: int
    extra: dict = field(default_factory=dict)


def cvar(values: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    """CVaR_alpha: media de la fracción alpha de menor energía (Barkoutsos et al., 2020)."""
    order = np.argsort(values)
    v, p = values[order], probs[order]
    cum = np.cumsum(p)
    k = np.searchsorted(cum, alpha)
    head = p[:k].sum()
    return float((v[:k] @ p[:k] + (alpha - head) * v[k]) / alpha)


def run_qaoa(qubo: QUBO, cost_fn, reps: int = 2, objective: str = "cvar", alpha: float = 0.1,
             restarts: int = 3, maxiter: int = 200, shots: int = 4096, seed: int = 7,
             optimal_cost: float | None = None, energies: np.ndarray | None = None,
             fixed_params=None) -> QAOAResult:
    rng = np.random.default_rng(seed)
    H = qubo.to_ising()
    nq = H.num_qubits
    ansatz = qaoa_circuit(H, reps)
    flat = ansatz
    ops = flat.count_ops()
    twoq = int(sum(v for k, v in ops.items() if k in ("rzz", "cx", "cz", "rzx")))
    E = qubo.all_energies() if energies is None else energies

    n_evals = 0

    # Los γ se optimizan normalizados (γ = γ' * scale) para que COBYLA trabaje en escala O(1)
    scale = 1.0 / max(1.0, float(np.abs(H.coeffs).max()))

    def to_theta(z):
        return np.concatenate([z[:reps], z[reps:] * scale])

    def objective_fn(z):
        nonlocal n_evals
        n_evals += 1
        probs = Statevector(ansatz.assign_parameters(to_theta(z))).probabilities()
        if objective == "cvar":
            return cvar(E, probs, alpha)
        return float(E @ probs)

    t0 = time.perf_counter()
    best = None
    if fixed_params is not None:  # transferencia de parámetros: sin optimización interna
        restarts = 0

        class _Fixed:
            x = np.asarray(fixed_params, float)
        best = _Fixed()
    for _ in range(restarts):
        # gamma pequeño (la escala de energía es grande) y beta en [0, pi/2]
        x0 = np.concatenate([rng.uniform(0, np.pi / 2, reps),   # β
                             rng.uniform(0, np.pi, reps)])      # γ' (normalizado)
        # parámetros ordenados como β[0..p-1], γ[0..p-1]
        res = minimize(objective_fn, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
        if best is None or res.fun < best.fun:
            best = res
    opt_s = time.perf_counter() - t0

    theta = to_theta(best.x)
    probs = Statevector(ansatz.assign_parameters(theta)).probabilities()
    expected = float(E @ probs)

    # Muestreo final con shots (como en un dispositivo)
    meas = ansatz.assign_parameters(theta)
    meas.measure_all()
    sampler = StatevectorSampler(seed=seed)
    counts = sampler.run([meas], shots=shots).result()[0].data.meas.get_counts()
    feas, opt_hits = 0, 0
    best_perm, best_cost, best_e = None, None, np.inf
    for bits, c in counts.items():
        x = np.array([int(b) for b in bits[::-1]])  # qubit 0 = carácter derecho
        e = float(qubo.energy(x)[0])
        best_e = min(best_e, e)
        perm = qubo.decode(x)
        if perm is None:
            continue
        feas += c
        cost = cost_fn(perm)
        if optimal_cost is not None and abs(cost - optimal_cost) < 1e-9:
            opt_hits += c
        if best_cost is None or cost < best_cost:
            best_perm, best_cost = perm, cost
    return QAOAResult(
        reps=reps, lam=qubo.lam, objective=objective, params=theta.tolist(), n_qubits=nq,
        n_evals=n_evals, opt_seconds=opt_s, shots=shots, feasible_fraction=feas / shots,
        prob_optimal=opt_hits / shots, best_perm=best_perm, best_cost=best_cost,
        best_energy_sampled=best_e, expected_energy=expected,
        circuit_depth=flat.depth(), two_qubit_gates=twoq,
        extra={"normalized_params": np.asarray(best.x).tolist()},
    )


def random_baseline(qubo: QUBO, cost_fn, shots: int, seed: int, optimal_cost=None):
    """Referencia: muestreo uniforme de bitstrings (lo que daría QAOA con parámetros nulos)."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(shots, qubo.nvars))
    feas = opt = 0
    best = None
    for x in X:
        p = qubo.decode(x)
        if p is None:
            continue
        feas += 1
        c = cost_fn(p)
        best = c if best is None else min(best, c)
        if optimal_cost is not None and abs(c - optimal_cost) < 1e-9:
            opt += 1
    return {"feasible_fraction": feas / shots, "prob_optimal": opt / shots, "best_cost": best}
