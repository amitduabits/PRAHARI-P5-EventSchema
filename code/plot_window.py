"""Figure for E5.5: the two-sided error curve and where each estimator lands."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results" / "p5_fusion.json"
OUT = ROOT.parent / "figs" / "p5_empirical_window.png"


def main() -> None:
    b = next(x for x in json.load(SRC.open())["results"]
             if x["experiment"] == "E5.5_empirical_window_model")
    rows = [r for r in b["rows"] if r["window_s"] > 0]
    w = [r["window_s"] for r in rows]
    red = [r["redundant_rate"] for r in rows]
    mask = [r["masked_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(w, red, marker="o", ms=3, label="redundant alerts per incident (false positive)")
    ax.plot(w, mask, marker="s", ms=3, label="incidents masked (false negative)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("collapse window W (s, log scale)")
    ax.set_ylabel("rate per incident (log scale)")
    ax.grid(alpha=0.3, which="both")

    est = b["estimators"]
    marks = [
        (est["debar_wespi"]["window_s"], "Debar-Wespi", "tab:green"),
        (est["kneedle_window_s"], "Kneedle", "tab:red"),
        (est["spread_rule"]["window_s"], "max+2sd", "tab:purple"),
        (est["cost_weighted"]["1.0:10.0"]["window_s"], "cost 1:10", "tab:brown"),
        (b["deployed_window_s"], "deployed 120 s", "black"),
    ]
    # Merge coincident estimators into one label: two of them landing on the
    # same second is the finding, not a plotting problem.
    merged: dict[float, list] = {}
    for x, label, col in marks:
        merged.setdefault(round(float(x), 2), []).append((label, col))
    marks = [(x, " = ".join(l for l, _ in v), v[0][1]) for x, v in sorted(merged.items())]

    lo, hi = ax.get_ylim()
    for i, (x, label, col) in enumerate(marks):
        ax.axvline(x, color=col, ls="--", lw=1.1, alpha=0.8)
        # Stagger the labels: four of the five estimators land within a few
        # seconds of each other, which is the result, but it makes them
        # unreadable stacked at the same height.
        y = hi * (10 ** (-0.45 * (i % 4)))
        ax.text(x, y, f" {label}", rotation=90, va="top", ha="left",
                fontsize=7.5, color=col)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Two-sided error against collapse window (SYNTHETIC, 12,000 incidents)",
                 fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
