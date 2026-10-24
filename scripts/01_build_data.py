"""Etapa 0: reconstruye F y D desde qap10.mps.gz y verifica los óptimos de control.

Uso:  python scripts/01_build_data.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from qap.mps_reader import reconstruct_qap  # noqa: E402
from qap.problem import QAP, brute_force  # noqa: E402


def main():
    F, D, M, info = reconstruct_qap(ROOT / "data" / "qap10.mps.gz")
    qap = QAP(F, D, "qap10")
    qap.save_csv(ROOT / "data")
    print("Linealización:", json.dumps(info, indent=1))
    print("F =\n", F)
    print("D =\n", D)

    # Comprobación de escala: el costo con el tensor original M debe coincidir con F*D
    p = np.random.default_rng(1).permutation(10)
    c_m = sum(M[i, k, p[i], p[k]] for i in range(10) for k in range(i + 1, 10))
    assert abs(c_m - qap.cost(p)) < 1e-9, "La escala de F*D no coincide con el MPS"

    results = {"linearization": info, "control": {}}
    for n, expected in [(3, 24), (4, 32), (10, 340)]:
        t0 = time.perf_counter()
        perm, cost = brute_force(qap.sub(n))
        dt = time.perf_counter() - t0
        ok = abs(cost - expected) < 1e-9
        print(f"N={n:2d}: óptimo={cost:.0f} (esperado {expected}) {'OK' if ok else 'ERROR'}"
              f"  asignación={perm}  tiempo={dt:.2f}s")
        results["control"][n] = {"cost": cost, "expected": expected, "perm": perm,
                                 "seconds": dt, "ok": ok}
        assert ok
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "00_data_check.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
