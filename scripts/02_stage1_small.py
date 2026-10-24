"""Etapa 1: subinstancias N=3 (9 qubits) y N=4 (16 qubits).

Compara: fuerza bruta (exacta), búsqueda local clásica, muestreo aleatorio y QAOA en Qiskit,
con barridos de profundidad p, del factor de penalización λ y de la función objetivo
(valor esperado vs CVaR).

Uso:  python scripts/02_stage1_small.py
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qap.classical import multistart_local_search  # noqa: E402
from qap.problem import QAP, brute_force  # noqa: E402
from qap.qaoa import random_baseline, run_qaoa  # noqa: E402
from qap.qubo import build_qubo  # noqa: E402

SEED = 7
SHOTS = 4096


def main():
    qap = QAP.from_csv(ROOT / "data/qap10_F.csv", ROOT / "data/qap10_D.csv", "qap10")
    out = {"seed": SEED, "shots": SHOTS, "instances": {}}
    for n in (3, 4):
        sub = qap.sub(n)
        t0 = time.perf_counter()
        perm, opt = brute_force(sub)
        bf_s = time.perf_counter() - t0
        ls = multistart_local_search(sub, starts=20, seed=SEED)
        rec = {"brute_force": {"perm": perm, "cost": opt, "seconds": bf_s},
               "local_search": {k: ls[k] for k in ("perm", "cost", "seconds", "starts")},
               "qaoa": []}
        configs = [(p, 1.0, "cvar") for p in (1, 2, 3)]
        if n == 4:
            configs += [(2, 0.25, "cvar"), (2, 0.5, "cvar"), (2, 1.0, "expectation")]
        else:
            configs += [(2, 0.25, "cvar"), (2, 0.5, "cvar"), (2, 1.0, "expectation")]
        base_q = build_qubo(sub.F, sub.D)
        rec["lambda_safe"] = base_q.lam
        rec["random"] = random_baseline(base_q, sub.cost, SHOTS, SEED, opt)
        for reps, lam_f, obj in configs:
            q = build_qubo(sub.F, sub.D, lam_factor=lam_f)
            r = run_qaoa(q, sub.cost, reps=reps, objective=obj, restarts=2 if n == 4 else 3,
                         maxiter=150, shots=SHOTS, seed=SEED, optimal_cost=opt)
            d = asdict(r)
            d["lambda_factor"] = lam_f
            rec["qaoa"].append(d)
            print(f"N={n} p={reps} λ={q.lam:.0f} ({lam_f}x) obj={obj:11s} "
                  f"factible={r.feasible_fraction:6.2%} P(ópt)={r.prob_optimal:6.2%} "
                  f"mejor={r.best_cost} t={r.opt_seconds:.1f}s evals={r.n_evals}")
        print(f"N={n}: óptimo={opt}  búsqueda local={ls['cost']}  "
              f"aleatorio factible={rec['random']['feasible_fraction']:.3%}")
        out["instances"][n] = rec
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/01_stage1.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
