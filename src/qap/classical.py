"""Métodos clásicos de referencia: búsqueda local por intercambios y recocido simulado."""
from __future__ import annotations

import math
import time

import numpy as np

from .problem import QAP


def swap_delta(qap: QAP, p: np.ndarray, r: int, s: int) -> float:
    """Cambio de costo O(n) al intercambiar las ubicaciones de los elementos r y s."""
    F, D = qap.F, qap.D
    pr, ps = p[r], p[s]
    mask = np.ones(qap.n, dtype=bool)
    mask[[r, s]] = False
    others = p[mask]
    return float(np.sum((F[r, mask] - F[s, mask]) * (D[ps, others] - D[pr, others])))


def local_search(qap: QAP, p0, rng=None):
    """Descenso por mejor mejora en el vecindario de intercambios (2-swap)."""
    p = np.array(p0)
    c = qap.cost(p)
    n = qap.n
    while True:
        best = (0.0, None)
        for r in range(n):
            for s in range(r + 1, n):
                d = swap_delta(qap, p, r, s)
                if d < best[0] - 1e-12:
                    best = (d, (r, s))
        if best[1] is None:
            return p, c
        r, s = best[1]
        p[r], p[s] = p[s], p[r]
        c += best[0]


def multistart_local_search(qap: QAP, starts: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best_p, best_c = None, math.inf
    history = []
    for _ in range(starts):
        p, c = local_search(qap, rng.permutation(qap.n))
        history.append(c)
        if c < best_c:
            best_p, best_c = p.copy(), c
    return {"perm": best_p.tolist(), "cost": best_c, "seconds": time.perf_counter() - t0,
            "starts": starts, "costs": history}


def simulated_annealing(qap: QAP, iters: int = 200_000, t0: float | None = None,
                        t_end: float = 0.05, seed: int = 0):
    """Recocido simulado con movimientos de intercambio (baseline clásico fuerte)."""
    rng = np.random.default_rng(seed)
    n = qap.n
    p = rng.permutation(n)
    c = qap.cost(p)
    best_p, best_c = p.copy(), c
    if t0 is None:
        t0 = float(np.mean(qap.F[qap.F > 0]) * np.mean(qap.D[qap.D > 0]))
    alpha = (t_end / t0) ** (1.0 / iters)
    T = t0
    start = time.perf_counter()
    for _ in range(iters):
        r, s = rng.choice(n, 2, replace=False)
        d = swap_delta(qap, p, r, s)
        if d <= 0 or rng.random() < math.exp(-d / T):
            p[r], p[s] = p[s], p[r]
            c += d
            if c < best_c - 1e-12:
                best_p, best_c = p.copy(), c
        T *= alpha
    return {"perm": best_p.tolist(), "cost": best_c, "seconds": time.perf_counter() - start,
            "iters": iters}
