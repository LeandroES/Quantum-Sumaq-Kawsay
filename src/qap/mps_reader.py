"""Reconstrucción de las matrices F (flujo) y D (distancia) a partir de qap10.mps.gz.

El archivo de MIPLIB contiene una linealización del QAP:
  * X001..X100  -> variables de asignación x_ij  (índice = 10*i + j + 1).
  * Filas NX01..NX10 / NX11..NX20 -> restricciones de asignación (filas / columnas).
  * Y???A???    -> variables auxiliares que reemplazan al producto x_ij * x_kl.
    Cada Y aparece en filas de enlace C??A### junto a exactamente dos variables X,
    y su coeficiente en la función objetivo (fila NOBJ) es F_ik * D_jl.

Procedimiento:
  1. Para cada Y se identifican las dos X con las que comparte filas de enlace.
  2. Se arma el tensor M[i,k,j,l] = coeficiente objetivo (con i<k, j!=l).
  3. Se verifica que la matriz {(i,k)} x {(j,l)} tenga rango 1 (es un producto F_ik * D_jl).
  4. Se factoriza con enteros: D = fila / mcd, F = columna / D.
La factorización tiene una ambigüedad de escala (F*c, D/c) que no cambia ningún costo.
"""
from __future__ import annotations

import collections
import gzip
import math
from pathlib import Path

import numpy as np


def read_mps(path: str | Path):
    """Devuelve (filas_de_restricción, coeficientes_objetivo) del archivo MPS."""
    rows: dict[str, dict[str, float]] = collections.defaultdict(dict)
    obj: dict[str, float] = {}
    section = None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.startswith(" "):
                section = line.split()[0]
                continue
            if section != "COLUMNS" or "'MARKER'" in line:
                continue
            tok = line.split()
            col = tok[0]
            for row, val in zip(tok[1::2], tok[2::2]):
                if row == "NOBJ":
                    obj[col] = float(val)
                else:
                    rows[row][col] = float(val)
    return rows, obj


def reconstruct_qap(path: str | Path, n: int = 10):
    """Reconstruye (F, D) enteros y simétricos y el tensor M de la linealización."""
    rows, obj = read_mps(path)
    xs = {c for r in rows.values() for c in r if c.startswith("X")}
    ys = {c for r in rows.values() for c in r if c.startswith("Y")} | set(obj)
    if len(xs) != n * n:
        raise ValueError(f"Se esperaban {n*n} variables X y hay {len(xs)}")
    xidx = {x: int(x[1:]) - 1 for x in xs}

    y_rows = collections.defaultdict(list)
    for r, d in rows.items():
        for c in d:
            if c.startswith("Y"):
                y_rows[c].append(r)

    M = np.zeros((n, n, n, n))
    for y in ys:
        linked = sorted({xidx[c] for r in y_rows[y] for c in rows[r] if c.startswith("X")})
        if len(linked) != 2:
            raise ValueError(f"{y} enlaza {len(linked)} variables X (se esperaban 2)")
        (i, j), (k, l) = divmod(linked[0], n), divmod(linked[1], n)
        M[i, k, j, l] += obj.get(y, 0.0)

    pairs = [(i, k) for i in range(n) for k in range(i + 1, n)]
    locs = [(j, l) for j in range(n) for l in range(n) if j != l]
    A = np.array([[M[i, k, j, l] for (j, l) in locs] for (i, k) in pairs])
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[1] > 1e-9 * sv[0]:
        raise ValueError("Los coeficientes no forman un producto F*D (rango > 1)")

    ref = int(np.argmax(A.max(axis=1)))
    g = math.gcd(*[int(round(v)) for v in A[ref] if v])
    dvec = A[ref] / g
    col = int(np.argmax(dvec))
    fvec = A[:, col] / dvec[col]
    F = np.zeros((n, n))
    D = np.zeros((n, n))
    for (i, k), v in zip(pairs, fvec):
        F[i, k] = F[k, i] = v
    for (j, l), v in zip(locs, dvec):
        D[j, l] = v
    err = float(np.abs(np.outer(fvec, dvec) - A).max())
    info = {
        "n_x": len(xs),
        "n_y": len(ys),
        "n_rows": len(rows),
        "singular_values": sv[:3].tolist(),
        "max_reconstruction_error": err,
        "D_symmetric": bool(np.allclose(D, D.T)),
    }
    return np.rint(F).astype(int), np.rint(D).astype(int), M, info
