"""Formulación QUBO / Ising del QAP (o de un subproblema con elementos fijos).

Variables binarias x[a] con a = i*m + j  ->  "el elemento i va a la ubicación j".

    H(x) = sum_{i<k} sum_{j!=l} F_ik D_jl x_ij x_kl          (costo cuadrático)
         + sum_{i,j} h_ij x_ij                                 (costo lineal, subproblemas)
         + lam * sum_i (sum_j x_ij - 1)^2 + lam * sum_j (sum_i x_ij - 1)^2

Con x = (1 - Z)/2 se obtiene un Hamiltoniano de Ising (SparsePauliOp de Qiskit).
Convención de bits: el qubit a corresponde a x[a]; en una bitstring de Qiskit
(little-endian) el carácter de más a la derecha es el qubit 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass
class QUBO:
    Q: np.ndarray        # matriz simétrica (m^2 x m^2); energía = x^T Q x + c (diag = lineal)
    const: float
    m: int
    lam: float

    @property
    def nvars(self) -> int:
        return self.m * self.m

    def energy(self, x: np.ndarray) -> np.ndarray:
        """Energía QUBO de una o varias asignaciones binarias (filas de x)."""
        x = np.atleast_2d(x).astype(float)
        return np.einsum("ni,ij,nj->n", x, self.Q, x) + self.const

    def all_energies(self) -> np.ndarray:
        """Energía de las 2^(m^2) bitstrings, indexadas como en un statevector de Qiskit."""
        nv = self.nvars
        idx = np.arange(2 ** nv, dtype=np.int64)
        X = ((idx[:, None] >> np.arange(nv)) & 1).astype(np.float32)
        Qf = self.Q.astype(np.float32)
        return np.einsum("ni,ij,nj->n", X, Qf, X, optimize=True).astype(float) + self.const

    def decode(self, x) -> list[int] | None:
        """Convierte bits a permutación; None si viola alguna restricción."""
        X = np.asarray(x).reshape(self.m, self.m)
        if not (np.all(X.sum(0) == 1) and np.all(X.sum(1) == 1)):
            return None
        return [int(np.argmax(r)) for r in X]

    def to_ising(self) -> SparsePauliOp:
        """Hamiltoniano de Ising equivalente (mismo espectro que la energía QUBO)."""
        nv = self.nvars
        terms: dict[str, float] = {}
        const = self.const

        def add(label_qubits, coeff):
            label = ["I"] * nv
            for q in label_qubits:
                label[nv - 1 - q] = "Z"
            key = "".join(label)
            terms[key] = terms.get(key, 0.0) + coeff

        for a in range(nv):
            # Q_aa x_a = Q_aa (1 - Z_a)/2
            const += self.Q[a, a] / 2
            add([a], -self.Q[a, a] / 2)
            for b in range(a + 1, nv):
                w = 2 * self.Q[a, b]  # x^T Q x incluye Q_ab y Q_ba
                if w == 0:
                    continue
                # w x_a x_b = w/4 (1 - Z_a - Z_b + Z_a Z_b)
                const += w / 4
                add([a], -w / 4)
                add([b], -w / 4)
                add([a, b], w / 4)
        terms["I" * nv] = terms.get("I" * nv, 0.0) + const
        items = [(k, v) for k, v in terms.items() if abs(v) > 1e-12]
        return SparsePauliOp.from_list(items).simplify()


def build_qubo(F: np.ndarray, D: np.ndarray, lam: float | None = None,
               linear: np.ndarray | None = None, lam_factor: float = 1.0) -> QUBO:
    """Construye el QUBO de un QAP m x m (opcionalmente con costo lineal h[i,j]).

    Si lam es None se usa una cota segura: lam > max_a sum_b |costo(a,b)| + |h_a|.
    Con esa cota, añadir o quitar un 1 en una asignación válida nunca reduce la
    energía, así que el mínimo global del QUBO es siempre una permutación factible.
    """
    m = F.shape[0]
    nv = m * m
    C = np.zeros((nv, nv))
    for i in range(m):
        for k in range(m):
            if i == k:
                continue
            for j in range(m):
                for l in range(m):
                    if j == l:
                        continue
                    # término F_ik D_jl x_ij x_kl repartido simétricamente (i<k contado una vez)
                    C[i * m + j, k * m + l] += 0.5 * F[i, k] * D[j, l]
    h = np.zeros(nv) if linear is None else np.asarray(linear, float).reshape(nv)
    if lam is None:
        lam = lam_factor * (float(np.max(2 * np.abs(C).sum(1) + np.abs(h))) + 1.0)
    Q = C.copy()
    Q[np.diag_indices(nv)] += h
    const = 0.0
    # lam * (sum_j x_ij - 1)^2 = lam*(sum_j x_ij + 2 sum_{j<l} x_ij x_il - 2 sum_j x_ij + 1)
    for i in range(m):
        vars_ = [i * m + j for j in range(m)]
        _add_onehot(Q, vars_, lam)
        const += lam
    for j in range(m):
        vars_ = [i * m + j for i in range(m)]
        _add_onehot(Q, vars_, lam)
        const += lam
    return QUBO(Q, const, m, lam)


def _add_onehot(Q, vars_, lam):
    for a in vars_:
        Q[a, a] += -lam
        for b in vars_:
            if b != a:
                Q[a, b] += lam  # simétrico: Q_ab + Q_ba = 2*lam por par


def perm_to_bits(p, m: int) -> np.ndarray:
    x = np.zeros(m * m, dtype=int)
    for i, j in enumerate(p):
        x[i * m + j] = 1
    return x
