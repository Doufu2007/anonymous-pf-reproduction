"""Audit and analyse paired metrics from a QM-HPF reproduction run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_rel, wilcoxon


METRICS = ("rmse", "mae", "mean_ess", "mean_distinct", "runtime_s")
METHODS = ("bootstrap", "mh", "qm_hpf", "rotation_only", "crossover_only")


def ci95(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sd, mean - half, mean + half


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"seed", "method", *METRICS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"raw metrics missing fields: {sorted(required)}")
    for row in rows:
        for metric in METRICS:
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError(f"non-finite {metric}: {row}")
    return rows


def paired(rows: list[dict[str, str]], left: str, right: str, metric: str) -> np.ndarray:
    l = {int(r["seed"]): float(r[metric]) for r in rows if r["method"] == left}
    r = {int(r["seed"]): float(r[metric]) for r in rows if r["method"] == right}
    if set(l) != set(r):
        raise ValueError(f"unpaired seeds for {left}/{right}: {sorted(set(l) ^ set(r))}")
    return np.array([l[seed] - r[seed] for seed in sorted(l)], dtype=float)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (len(ordered) - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def bootstrap_ci(values: np.ndarray, draws: int = 10000, seed: int = 20261008) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample = rng.choice(np.asarray(values, dtype=float), size=(draws, len(values)), replace=True)
    means = np.mean(sample, axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_paired_plot(rows: list[dict[str, str]], out_path: Path) -> None:
    seed_order = sorted({int(r["seed"]) for r in rows})
    by_method = {
        method: {int(r["seed"]): float(r["rmse"]) for r in rows if r["method"] == method}
        for method in METHODS
    }
    comparisons = [("bootstrap", "Bootstrap PF"), ("crossover_only", "Crossover-only")]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    ax = axes[0]
    for comparator, label in comparisons:
        x = np.array([by_method[comparator][s] for s in seed_order])
        y = np.array([by_method["qm_hpf"][s] for s in seed_order])
        ax.plot(x, y, "o", ms=3.5, alpha=0.65, label=f"QM-HPF vs {label}")
    limits = ax.get_xlim() + ax.get_ylim()
    lo = min(limits)
    hi = max(limits)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("Comparator RMSE")
    ax.set_ylabel("QM-HPF RMSE")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Per-seed RMSE")

    ax = axes[1]
    positions = np.arange(len(comparisons))
    for pos, (comparator, label) in zip(positions, comparisons):
        diff = np.array([by_method["qm_hpf"][s] - by_method[comparator][s] for s in seed_order])
        mean = float(np.mean(diff))
        low, high = bootstrap_ci(diff)
        ax.scatter(np.full_like(diff, pos, dtype=float), diff, s=12, alpha=0.55)
        ax.errorbar(pos, mean, yerr=[[mean - low], [high - mean]], fmt="o", color="black", capsize=4)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(positions, [label.replace(" PF", "") for _, label in comparisons], rotation=20, ha="right")
    ax.set_ylabel("QM-HPF minus comparator RMSE")
    ax.set_title("Paired differences (bootstrap 95% CI)")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.run_dir / "raw_metrics.csv")
    by_method = {method: [r for r in rows if r["method"] == method] for method in METHODS}
    if any(len(values) < 2 for values in by_method.values()):
        raise ValueError("every method needs at least two runs")

    summary: dict[str, dict[str, float | int]] = {}
    for method, values in by_method.items():
        summary[method] = {"runs": len(values)}
        for metric in METRICS:
            mean, sd, low, high = ci95(np.array([float(row[metric]) for row in values]))
            summary[method][f"{metric}_mean"] = mean
            summary[method][f"{metric}_sd"] = sd
            summary[method][f"{metric}_ci95_low"] = low
            summary[method][f"{metric}_ci95_high"] = high

    comparisons: dict[str, dict[str, float | int | str]] = {}
    pvalues: dict[str, float] = {}
    for comparator in ("bootstrap", "mh", "rotation_only", "crossover_only"):
        name = f"qm_hpf_minus_{comparator}"
        diff = paired(rows, "qm_hpf", comparator, "rmse")
        mean, sd, low, high = ci95(diff)
        test = ttest_rel(
            [float(r["rmse"]) for r in by_method["qm_hpf"]],
            [float(r["rmse"]) for r in by_method[comparator]],
        )
        try:
            signed = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
            wilcoxon_p = float(signed.pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        pvalues[name] = float(test.pvalue)
        comparisons[name] = {
            "n": len(diff),
            "direction": "negative means QM-HPF has lower RMSE",
            "mean_difference": mean,
            "sd_difference": sd,
            "ci95_low": low,
            "ci95_high": high,
            "paired_t_p": float(test.pvalue),
            "wilcoxon_p": wilcoxon_p,
            "cohen_dz": float(mean / sd) if sd > 0 else 0.0,
            "qm_hpf_wins": int(np.sum(diff < 0)),
            "qm_hpf_win_rate": float(np.mean(diff < 0)),
            "bootstrap_ci95_low": bootstrap_ci(diff)[0],
            "bootstrap_ci95_high": bootstrap_ci(diff)[1],
        }
    adjusted = holm_adjust(pvalues)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value

    audit = {
        "raw_rows": len(rows),
        "expected_rows": len(METHODS) * len(by_method["bootstrap"]),
        "methods": {method: len(values) for method, values in by_method.items()},
        "finite_metrics": True,
        "paired_seed_sets": True,
    }
    result = {"run_dir": str(args.run_dir), "audit": audit, "summary": summary, "paired_rmse": comparisons,
              "multiple_comparison_note": "Four predeclared QM-HPF RMSE comparisons use Holm adjustment; this does not turn a non-significant result into evidence of equivalence."}
    (args.run_dir / "paired_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_paired_plot(rows, args.run_dir / "paired_rmse.png")

    lines = ["# Paired analysis", "", f"Run directory: `{args.run_dir}`", "", "## Aggregate RMSE", "",
             "| Method | Mean | SD | 95% CI |", "|---|---:|---:|---:|"]
    for method in METHODS:
        item = summary[method]
        lines.append(f"| {method} | {item['rmse_mean']:.4f} | {item['rmse_sd']:.4f} | [{item['rmse_ci95_low']:.4f}, {item['rmse_ci95_high']:.4f}] |")
    lines += ["", "## Paired RMSE differences", "", "Negative values favour QM-HPF.", "",
              "| Comparison | Mean difference | 95% CI | Bootstrap CI | Cohen dz | Wins | Paired t p | Wilcoxon p | Holm p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in comparisons.items():
        lines.append(f"| {name} | {item['mean_difference']:.4f} | [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] | [{item['bootstrap_ci95_low']:.4f}, {item['bootstrap_ci95_high']:.4f}] | {item['cohen_dz']:.3f} | {item['qm_hpf_wins']}/{item['n']} | {item['paired_t_p']:.4g} | {item['wilcoxon_p']:.4g} | {item['holm_adjusted_p']:.4g} |")
    lines += ["", "The paired tests quantify uncertainty across the fixed seed set; they do not establish equivalence or hardware acceleration.", ""]
    (args.run_dir / "paired_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
