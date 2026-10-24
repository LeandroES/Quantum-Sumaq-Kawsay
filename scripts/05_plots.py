"""Genera las figuras de results/ a partir de los JSON de las etapas."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def stage1():
    d = json.loads((RES / "01_stage1.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (n, rec) in zip(axes, d["instances"].items()):
        rows = [q for q in rec["qaoa"] if q["lambda_factor"] == 1.0 and q["objective"] == "cvar"]
        labels = ["aleatorio"] + [f"QAOA p={q['reps']}" for q in rows]
        feas = [rec["random"]["feasible_fraction"]] + [q["feasible_fraction"] for q in rows]
        popt = [rec["random"]["prob_optimal"]] + [q["prob_optimal"] for q in rows]
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], [100 * f for f in feas], 0.4, label="muestras factibles")
        ax.bar([i + 0.2 for i in x], [100 * p for p in popt], 0.4, label="muestras óptimas")
        ax.set_xticks(list(x), labels)
        ax.set_ylabel("% de shots")
        nq = int(n) ** 2
        ax.set_title(f"N={n} ({nq} qubits), óptimo={rec['brute_force']['cost']:.0f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "fig1_stage1_qaoa.png", dpi=150)


def stage2():
    d = json.loads((RES / "02_stage2_n10.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label, color in [("lns_exact", "LNS exacto (clásico)", "tab:gray"),
                              ("qaoa_lns_transfer", "QAOA-LNS (Qiskit)", "tab:purple")]:
        for i, r in enumerate(d["runs"][key]):
            ax.plot(range(1, len(r["best_trace"]) + 1), r["best_trace"], color=color, alpha=0.7, label=label if i == 0 else None)
    ax.axhline(340, ls="--", color="k", lw=1, label="óptimo 340")
    ax.set_xscale("log")
    ax.set_xlim(1, None)
    ax.set_xlabel("iteración (subproblemas de 9 qubits)")
    ax.set_ylabel("mejor costo factible")
    ax.set_title("qap10 (N=10): convergencia por semilla")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "fig2_stage2_convergence.png", dpi=150)


def scaling():
    p = RES / "03_scaling.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    ns = [r["n"] for r in d["instances"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in d["instances"][0]["methods"]:
        axes[0].plot(ns, [r["methods"][m]["mean_seconds"] for r in d["instances"]], "o-", label=m)
        axes[1].plot(ns, [r["methods"][m]["mean_gap_vs_best_pct"] for r in d["instances"]], "o-", label=m)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("tiempo medio por corrida (s)")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("gap medio vs mejor conocido (%)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "fig3_scaling.png", dpi=150)


if __name__ == "__main__":
    stage1()
    stage2()
    scaling()
    print("Figuras en", RES)
