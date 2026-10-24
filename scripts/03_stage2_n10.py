"""Etapa 2: instancia completa qap10 (N = 10, óptimo de referencia 340).

Métodos comparados (todos producen solo asignaciones válidas):
  * Fuerza bruta (referencia exacta, 10! permutaciones en lotes).
  * Búsqueda local multiarranque (2-swap)            -> clásico.
  * Recocido simulado                                -> clásico.
  * LNS iterado con subproblemas exactos (m = 3)     -> control clásico del híbrido.
  * QAOA-LNS (m = 3, 9 qubits por subproblema)       -> híbrido con Qiskit.
      - parámetros transferidos (entrenados en 6 subproblemas, sin optimización interna)
      - parámetros optimizados en cada subproblema (corrida corta, para medir el costo)

Uso:  python scripts/03_stage2_n10.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from qap.classical import multistart_local_search, simulated_annealing  # noqa: E402
from qap.hybrid import lns, train_transfer_params  # noqa: E402
from qap.problem import QAP, brute_force, gap  # noqa: E402

OPT = 340
SEEDS = list(range(10))
ITERS = 3000


def summarize(name, runs):
    costs = [r["cost"] for r in runs]
    hits = []
    for r in runs:
        tr = r.get("best_trace")
        hits.append(next((i for i, c in enumerate(tr) if c <= OPT), None) if tr else None)
    s = {
        "method": name,
        "best_cost": float(min(costs)),
        "mean_cost": float(np.mean(costs)),
        "best_gap_pct": gap(min(costs), OPT),
        "mean_gap_pct": gap(float(np.mean(costs)), OPT),
        "runs_reaching_opt": int(sum(abs(c - OPT) < 1e-9 for c in costs)),
        "n_runs": len(runs),
        "mean_seconds": float(np.mean([r["seconds"] for r in runs])),
        "iters_to_opt": hits,
    }
    print(f"{name:32s} mejor={s['best_cost']:.0f} media={s['mean_cost']:.1f} "
          f"gap_medio={s['mean_gap_pct']:.2f}% ópt={s['runs_reaching_opt']}/{len(runs)} "
          f"t={s['mean_seconds']:.2f}s")
    return s


def main():
    qap = QAP.from_csv(ROOT / "data/qap10_F.csv", ROOT / "data/qap10_D.csv", "qap10")
    out = {"opt_reference": OPT, "seeds": SEEDS, "iters": ITERS, "summary": [], "runs": {}}

    t0 = time.perf_counter()
    perm, c = brute_force(qap)
    out["brute_force"] = {"perm": perm, "cost": c, "seconds": time.perf_counter() - t0}
    print(f"Fuerza bruta: {c:.0f} en {out['brute_force']['seconds']:.1f}s  {perm}")

    runs = [multistart_local_search(qap, starts=50, seed=s) for s in SEEDS]
    out["summary"].append(summarize("Búsqueda local 2-swap (50 arr.)", runs))
    out["runs"]["local_search"] = [{k: r[k] for k in ("perm", "cost", "seconds")} for r in runs]

    runs = [simulated_annealing(qap, iters=50_000, seed=s) for s in SEEDS]
    out["summary"].append(summarize("Recocido simulado (50k it.)", runs))
    out["runs"]["simulated_annealing"] = runs

    runs = [lns(qap, "exact", m=3, iters=ITERS, seed=s) for s in SEEDS]
    out["summary"].append(summarize("LNS exacto m=3 (control)", runs))
    out["runs"]["lns_exact"] = [{k: v for k, v in r.items() if k != "trace"} for r in runs]

    t = time.perf_counter()
    z = train_transfer_params(qap, m=3, reps=2, n_train=6, seed=11)
    out["qaoa_transfer_params"] = {"normalized_params": z, "train_seconds": time.perf_counter() - t}
    print(f"Parámetros QAOA transferidos (β, γ'): {np.round(z, 3).tolist()}")

    runs = [lns(qap, "qaoa", m=3, iters=ITERS, seed=s, qaoa_params=z, shots=512) for s in SEEDS]
    out["summary"].append(summarize("QAOA-LNS m=3 (param. transf.)", runs))
    for r in runs:
        r["qaoa_exact_hit_rate"] = r["qaoa_exact_hits"] / max(1, r["qaoa_calls"])
    out["runs"]["qaoa_lns_transfer"] = [{k: v for k, v in r.items()
                                         if k not in ("trace", "feasible_fractions")} for r in runs]
    print("  tasa de subproblemas resueltos al óptimo por QAOA:",
          [round(r["qaoa_exact_hit_rate"], 3) for r in runs],
          " fracción factible media:", [round(r["mean_feasible_fraction"], 3) for r in runs])

    runs = [lns(qap, "qaoa", m=3, iters=150, seed=s, shots=512) for s in SEEDS[:2]]
    out["summary"].append(summarize("QAOA-LNS m=3 (optimizado, 150 it.)", runs))
    for r in runs:
        r["qaoa_exact_hit_rate"] = r["qaoa_exact_hits"] / max(1, r["qaoa_calls"])
    out["runs"]["qaoa_lns_optimized"] = [{k: v for k, v in r.items()
                                          if k not in ("trace", "feasible_fractions")} for r in runs]

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/02_stage2_n10.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
