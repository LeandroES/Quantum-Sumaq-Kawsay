"""Definición del QAP y evaluación de costos.

Convención (la del archivo MIPLIB): cada par de elementos se cuenta una sola vez,
    C(p) = sum_{i<k} F[i,k] * D[p[i], p[k]]
donde p[i] es la ubicación asignada al elemento i (una permutación).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class QAP:
    F: np.ndarray
    D: np.ndarray
    name: str = "qap"

    @property
    def n(self) -> int:
        return self.F.shape[0]

    def cost(self, p) -> float:
        p = np.asarray(p)
        # F simétrica con diagonal nula -> la mitad de la suma completa cuenta cada par una vez
        return 0.5 * float(np.sum(self.F * self.D[np.ix_(p, p)]))

    def cost_batch(self, P: np.ndarray) -> np.ndarray:
        """Costo de muchas permutaciones (filas de P) a la vez."""
        out = np.zeros(P.shape[0])
        for i in range(self.n):
            for k in range(i + 1, self.n):
                f = self.F[i, k]
                if f:
                    out += f * self.D[P[:, i], P[:, k]]
        return out

    def sub(self, n: int) -> "QAP":
        """Subinstancia con las primeras n filas y columnas (Etapa 1 del reto)."""
        return QAP(self.F[:n, :n].copy(), self.D[:n, :n].copy(), f"{self.name}_N{n}")

    def is_permutation(self, p) -> bool:
        return sorted(p) == list(range(self.n))

    def save_csv(self, folder: str | Path):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savetxt(folder / f"{self.name}_F.csv", self.F, fmt="%d", delimiter=",")
        np.savetxt(folder / f"{self.name}_D.csv", self.D, fmt="%d", delimiter=",")

    @staticmethod
    def from_csv(f_path, d_path, name="qap") -> "QAP":
        F = np.loadtxt(f_path, delimiter=",", dtype=int)
        D = np.loadtxt(d_path, delimiter=",", dtype=int)
        return QAP(F, D, name)


def grid_instance(n: int, seed: int = 0, rows: int | None = None, density: float = 0.55) -> QAP:
    """Instancia sintética al estilo de qap10 (Nugent): ubicaciones en cuadrícula con
    distancia Manhattan y flujos enteros dispersos. Sirve para la prueba de escalado
    cuando no se dispone de las instancias ampliadas oficiales."""
    rng = np.random.default_rng(seed)
    rows = rows or int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    pts = [(r, c) for r in range(rows) for c in range(cols)][:n]
    D = np.array([[abs(a[0] - b[0]) + abs(a[1] - b[1]) for b in pts] for a in pts])
    F = np.zeros((n, n), dtype=int)
    choices = np.array([2, 4, 6, 8, 10, 12, 20])
    for i in range(n):
        for k in range(i + 1, n):
            if rng.random() < density:
                F[i, k] = F[k, i] = rng.choice(choices)
    return QAP(F, D, f"grid{n}_s{seed}")


def brute_force(qap: QAP, batch: int = 200_000):
    """Enumera las n! permutaciones en lotes, sin guardarlas todas en memoria."""
    best_c, best_p = np.inf, None
    it = itertools.permutations(range(qap.n))
    while True:
        chunk = list(itertools.islice(it, batch))
        if not chunk:
            break
        P = np.array(chunk, dtype=np.int16)
        c = qap.cost_batch(P)
        idx = int(np.argmin(c))
        if c[idx] < best_c:
            best_c, best_p = float(c[idx]), P[idx].tolist()
    return best_p, best_c


def gap(cost: float, opt: float) -> float:
    return 100.0 * (cost - opt) / opt
