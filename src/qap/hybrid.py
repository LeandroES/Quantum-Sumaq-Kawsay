"""Método híbrido QAOA-LNS (Large Neighborhood Search) para N = 10 y más.

Idea: no se puede simular un circuito de N^2 qubits, pero sí uno de m^2 qubits (m = 3 -> 9).
En cada iteración se "liberan" m elementos y las m ubicaciones que ocupan; el resto queda fijo.
El subproblema es un QAP m x m con un término lineal que recoge la interacción con los
elementos fijos:
    h[a, b] = sum_{k fijo} F[S_a, k] * D[L_b, p(k)]
Ese subproblema se convierte en QUBO -> Ising y se resuelve con QAOA en Qiskit.
Si la mejor muestra factible mejora el costo total, se acepta.

Para medir el aporte real de QAOA, cada subproblema se resuelve también por enumeración
exacta (m! permutaciones) y se registra si QAOA encontró el subóptimo exacto.
La variante "exacta" (solo clásica) usa el mismo esquema LNS sustituyendo QAOA por la
enumeración: es el control que permite atribuir diferencias al solver del subproblema.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from .problem import QAP
from .qaoa import run_qaoa
from .qubo import build_qubo


def subproblem(qap: QAP, p: np.ndarray, S: list[int]):
    """Devuelve (F_sub, D_sub, h, L, const) para liberar los elementos S."""
    S = list(S)
    L = [int(p[i]) for i in S]
    fixed = [k for k in range(qap.n) if k not in S]
    Fs = qap.F[np.ix_(S, S)]
    Ds = qap.D[np.ix_(L, L)]
    h = np.zeros((len(S), len(S)))
    for a, i in enumerate(S):
        for b, loc in enumerate(L):
            h[a, b] = sum(qap.F[i, k] * qap.D[loc, p[k]] for k in fixed)
    # costo de los pares fijo-fijo (no depende de la decisión)
    const = sum(qap.F[i, k] * qap.D[p[i], p[k]] for x, i in enumerate(fixed) for k in fixed[x + 1:])
    return Fs, Ds, h, L, const


def sub_cost(Fs, Ds, h, sigma):
    m = len(sigma)
    c = sum(Fs[a, b] * Ds[sigma[a], sigma[b]] for a in range(m) for b in range(a + 1, m))
    return c + sum(h[a, sigma[a]] for a in range(m))


def exact_sub(Fs, Ds, h):
    m = Fs.shape[0]
    return min(((sub_cost(Fs, Ds, h, s), list(s)) for s in itertools.permutations(range(m))))


def choose_subset(qap: QAP, rng, m: int) -> list[int]:
    """Semilla aleatoria + elementos con más flujo hacia ella (vecindario con interacción)."""
    seed = int(rng.integers(qap.n))
    w = qap.F[seed].astype(float) + 1e-3
    w[seed] = 0
    others = rng.choice(qap.n, size=m - 1, replace=False, p=w / w.sum())
    return [seed, *[int(o) for o in others]]


def lns(qap: QAP, solver: str = "qaoa", m: int = 3, iters: int = 60, seed: int = 0,
        reps: int = 2, shots: int = 1024, p0=None, log_every: int = 0,
        qaoa_params=None, patience: int = 25, kick: int = 3):
    """LNS iterado. solver = "qaoa" (Qiskit) o "exact" (enumeración m!, control clásico).

    qaoa_params: parámetros normalizados (β, γ') transferidos; si es None se optimizan
    en cada subproblema (más lento). Tras `patience` iteraciones sin mejorar el mejor
    costo se aplica una perturbación de `kick` intercambios aleatorios (ILS).
    """
    rng = np.random.default_rng(seed)
    qrng = np.random.default_rng(seed + 1_000_003)  # semillas de QAOA separadas: misma secuencia de vecindarios
    p = np.array(p0) if p0 is not None else rng.permutation(qap.n)
    cost = qap.cost(p)
    start_cost = cost
    best_p, best_cost = p.copy(), cost
    since_best = 0
    t0 = time.perf_counter()
    stats = {"qaoa_calls": 0, "qaoa_exact_hits": 0, "qaoa_no_feasible": 0,
             "accepted": 0, "kicks": 0, "feasible_fractions": [], "trace": [cost],
             "best_trace": [best_cost]}
    for it in range(iters):
        S = choose_subset(qap, rng, m)
        Fs, Ds, h, L, _ = subproblem(qap, p, S)
        current = list(range(m))  # identidad = asignación actual
        cur_val = sub_cost(Fs, Ds, h, current)
        best_val, best_sigma = exact_sub(Fs, Ds, h)
        if solver == "qaoa":
            q = build_qubo(Fs, Ds, linear=h)
            r = run_qaoa(q, lambda s: sub_cost(Fs, Ds, h, s), reps=reps, objective="cvar",
                         restarts=1, maxiter=60, shots=shots, seed=int(qrng.integers(1 << 30)),
                         optimal_cost=best_val, fixed_params=qaoa_params)
            stats["qaoa_calls"] += 1
            stats["feasible_fractions"].append(r.feasible_fraction)
            if r.best_perm is None:
                stats["qaoa_no_feasible"] += 1
                cand_val, cand = cur_val, current
            else:
                cand_val, cand = r.best_cost, r.best_perm
            if abs(cand_val - best_val) < 1e-9:
                stats["qaoa_exact_hits"] += 1
        else:
            cand_val, cand = best_val, best_sigma
        # Se aceptan mejoras y movimientos laterales (mismo costo) para no estancarse en mesetas
        if cand_val < cur_val - 1e-9 or (abs(cand_val - cur_val) < 1e-9 and list(cand) != current):
            stats["accepted"] += int(cand_val < cur_val - 1e-9)
            newp = p.copy()
            for a, i in enumerate(S):
                newp[i] = L[cand[a]]
            assert sorted(newp.tolist()) == list(range(qap.n)), "asignación inválida"
            p = newp
            cost = qap.cost(p)
        if cost < best_cost - 1e-9:
            best_p, best_cost, since_best = p.copy(), cost, 0
        else:
            since_best += 1
        if since_best >= patience:  # perturbación desde la mejor solución conocida
            p = best_p.copy()
            for _ in range(kick):
                a, b = rng.choice(qap.n, 2, replace=False)
                p[a], p[b] = p[b], p[a]
            cost = qap.cost(p)
            since_best = 0
            stats["kicks"] += 1
        stats["trace"].append(cost)
        stats["best_trace"].append(best_cost)
        if log_every and (it + 1) % log_every == 0:
            print(f"  [{solver}] iter {it+1}/{iters} mejor={best_cost:.0f}")
    stats.update({"perm": best_p.tolist(), "cost": best_cost, "start_cost": start_cost,
                  "seconds": time.perf_counter() - t0, "solver": solver, "m": m,
                  "iters": iters, "seed": seed})
    if stats["feasible_fractions"]:
        stats["mean_feasible_fraction"] = float(np.mean(stats["feasible_fractions"]))
    return stats


def train_transfer_params(qap: QAP, m: int = 3, reps: int = 2, n_train: int = 8, seed: int = 0):
    """Optimiza QAOA en varios subproblemas de entrenamiento y devuelve la mediana de los
    parámetros normalizados (transferencia de parámetros entre instancias similares)."""
    rng = np.random.default_rng(seed)
    zs = []
    for _ in range(n_train):
        p = rng.permutation(qap.n)
        S = choose_subset(qap, rng, m)
        Fs, Ds, h, _, _ = subproblem(qap, p, S)
        q = build_qubo(Fs, Ds, linear=h)
        best_val, _ = exact_sub(Fs, Ds, h)
        r = run_qaoa(q, lambda s: sub_cost(Fs, Ds, h, s), reps=reps, objective="cvar",
                     restarts=2, maxiter=120, shots=256, seed=int(rng.integers(1 << 30)),
                     optimal_cost=best_val)
        zs.append(r.extra["normalized_params"])
    return np.median(np.array(zs), axis=0).tolist()
