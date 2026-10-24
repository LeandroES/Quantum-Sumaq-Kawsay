"""Etapa 3: prueba de escalabilidad (N >= 14).

Por defecto usa instancias sintéticas al estilo qap10 (cuadrícula Manhattan + flujos
enteros dispersos). Para las instancias oficiales de la final, pasar las matrices:
    python scripts/04_scaling.py --F ruta/F.csv --D ruta/D.csv --name qapNN

Mide para cada método: mejor costo, brecha frente a la mejor solución encontrada por
cualquier método (no se conoce el óptimo), tiempo y memoria pico del proceso (RSS).
El componente cuántico (QAOA, 9 qubits por subproblema) no crece con N: lo que crece
es el número de subproblemas necesarios.
"""
import argparse
import json
import sys
import time
import resource
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from qap.classical import multistart_local_search, simulated_annealing  # noqa: E402
from qap.hybrid import lns, train_transfer_params  # noqa: E402
from qap.problem import QAP, grid_instance  # noqa: E402


def measured(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    dt = time.perf_counter() - t
    # memoria pico del proceso (RSS); tracemalloc distorsionaba mucho los tiempos
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return r, dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--F")
    ap.add_argument("--D")
    ap.add_argument("--name", default="custom")
    ap.add_argument("--sizes", default="10,12,14,16")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    if args.F:
        instances = [QAP.from_csv(args.F, args.D, args.name)]
    else:
        instances = [grid_instance(int(n), seed=0) for n in args.sizes.split(",")]

    qap10 = QAP.from_csv(ROOT / "data/qap10_F.csv", ROOT / "data/qap10_D.csv", "qap10")
    z = train_transfer_params(qap10, m=3, reps=2, n_train=6, seed=11)

    out = {"transfer_params": z, "instances": []}
    for qap in instances:
        n = qap.n
        iters = 150 * n
        rec = {"name": qap.name, "n": n, "qubo_vars_full": n * n,
               "statevector_bytes_full": 16 * 2.0 ** (n * n),
               "statevector_bytes_subproblem": 16 * 2 ** 9, "methods": {}}
        methods = {
            "local_search": lambda s: multistart_local_search(qap, starts=50, seed=s),
            "simulated_annealing": lambda s: simulated_annealing(qap, iters=20_000 * n // 10, seed=s),
            "lns_exact_m3": lambda s: lns(qap, "exact", m=3, iters=iters, seed=s),
            "qaoa_lns_m3": lambda s: lns(qap, "qaoa", m=3, iters=iters, seed=s, qaoa_params=z, shots=512),
        }
        for name, fn in methods.items():
            costs, times, mems = [], [], []
            for s in range(args.seeds):
                r, dt, mem = measured(fn, s)
                costs.append(r["cost"])
                times.append(dt)
                mems.append(mem)
            rec["methods"][name] = {"costs": costs, "best": min(costs), "mean": float(np.mean(costs)),
                                    "mean_seconds": float(np.mean(times)),
                                    "peak_mem_MiB": float(max(mems))}
        best = min(m["best"] for m in rec["methods"].values())
        rec["best_known"] = best
        for name, m in rec["methods"].items():
            m["mean_gap_vs_best_pct"] = 100 * (m["mean"] - best) / best
            print(f"N={n:2d} {name:20s} mejor={m['best']:.0f} media={m['mean']:.1f} "
                  f"gap={m['mean_gap_vs_best_pct']:.2f}% t={m['mean_seconds']:.2f}s "
                  f"RSS={m['peak_mem_MiB']:.0f}MiB")
        out["instances"].append(rec)
    (ROOT / "results").mkdir(exist_ok=True)
    fname = f"03_scaling_{args.name}.json" if args.F else "03_scaling.json"
    (ROOT / "results" / fname).write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
