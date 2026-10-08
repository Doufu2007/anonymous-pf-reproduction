"""Frozen two-contrast paired RMSE analysis of the exploratory supplement."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_1samp

from budget_benchmark import METHODS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confidence(values):
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return {"mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    adjusted, prior = {}, 0.0
    for rank, name in enumerate(order):
        prior = max(prior, min(1.0, (len(order) - rank) * pvalues[name]))
        adjusted[name] = prior
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    plan = json.loads((run / "frozen_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or manifest["completed_runs"] != 90:
        raise ValueError("incomplete or failed run bundle")
    if sha(run / "raw_metrics.csv") != manifest["raw_metrics_sha256"]:
        raise ValueError("raw results hash mismatch")
    with (run / "raw_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {method: {} for method in METHODS}
    for row in rows:
        seed, method = int(row["seed"]), row["method"]
        if seed in by_method[method]:
            raise ValueError("duplicate method/seed")
        by_method[method][seed] = row
    seeds = plan["seeds"]
    if any(sorted(by_method[method]) != seeds for method in METHODS):
        raise ValueError("missing/extra seeds or unpaired comparison")
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in ("rmse", "mae", "mean_ess", "mean_distinct", "runtime_s", "ess_per_second", "likelihood_evals", "budget_shortfall_steps"):
            values = np.array([float(by_method[method][seed][metric]) for seed in seeds])
            if not np.isfinite(values).all():
                raise ValueError("non-finite result")
            summary[method][metric] = confidence(values)
    comparisons, pvalues = {}, {}
    for comparator in ("bootstrap_200", "resample_move_100"):
        diff = np.array([float(by_method["qm_hpf_like_100"][seed]["rmse"])
                        - float(by_method[comparator][seed]["rmse"]) for seed in seeds])
        item = confidence(diff)
        bootstrap_rng = np.random.default_rng(plan["bootstrap_ci_seed"])
        bootstrap_means = bootstrap_rng.choice(diff, size=(plan["bootstrap_ci_draws"], len(diff)), replace=True).mean(axis=1)
        item.update(n=len(diff), bootstrap_ci95=[float(v) for v in np.quantile(bootstrap_means, [0.025, 0.975])],
                    paired_t_p=float(ttest_1samp(diff, popmean=0.0).pvalue),
                    cohen_dz=float(diff.mean() / diff.std(ddof=1)), qm_like_wins=int(np.sum(diff < 0)),
                    paired_differences=diff.tolist())
        name = "qm_hpf_like_100_minus_" + comparator
        comparisons[name] = item
        pvalues[name] = item["paired_t_p"]
    for name, p in holm(pvalues).items():
        comparisons[name]["holm_adjusted_p"] = p
    result = {"analysis_mode": "exploratory supplement designed after viewing native-cost results",
              "computed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "plan_locked_at": plan["locked_at"], "run_started_at": manifest["started_at"],
              "run_ended_at": manifest["ended_at"], "n_paired_seeds": len(seeds), "missingness": "none; no exclusions",
              "unit": "independently generated trajectory seed, shared observations within seed",
              "primary_metric": "trajectory RMSE of pre-move filtering estimate",
              "family": "two paired QM-HPF-like contrasts; Holm FWER correction",
              "raw_metrics_sha256": sha(run / "raw_metrics.csv"), "manifest_sha256": sha(run / "manifest.json"),
              "analysis_code_sha256": sha(Path(__file__)), "summary": summary, "paired_rmse": comparisons,
              "budget": "200 scalar likelihood evaluations per step upper cap; exact expenditure must be checked",
              "limitations": ["One source-inspired Gaussian synthetic model; no cross-task generalization.",
                  "Equal likelihood calls do not match wall-clock, particles, memory or transition-density operations.",
                  "QM-HPF-like kernel is heuristic and not established posterior invariant; not author-code reproduction.",
                  "Failure to reject a difference is not evidence of equivalence or noninferiority.",
                  "The resample-move step size is fixed, untuned; one MH step does not imply convergence."]}
    (run / "paired_analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Fair likelihood-cap exploratory supplement", "", result["analysis_mode"], "",
             "| Method | N | RMSE mean +/- SD | MAE mean | ESS | Distinct | Runtime seconds | Likelihood calls |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        v = summary[method]
        lines.append(f"| {method} | {200 if method == 'bootstrap_200' else 100} | {v['rmse']['mean']:.6f} +/- {v['rmse']['sd']:.6f} | {v['mae']['mean']:.6f} | {v['mean_ess']['mean']:.3f} | {v['mean_distinct']['mean']:.3f} | {v['runtime_s']['mean']:.4f} | {v['likelihood_evals']['mean']:.0f} |")
    lines += ["", "Negative paired differences favour QM-HPF-like.", "", "| Contrast | Mean difference | t 95% CI | Bootstrap 95% CI | dz | p | Holm p | Wins |", "|---|---:|---|---|---:|---:|---:|---:|"]
    for name, v in comparisons.items():
        lines.append(f"| {name} | {v['mean']:+.6f} | [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}] | [{v['bootstrap_ci95'][0]:+.6f}, {v['bootstrap_ci95'][1]:+.6f}] | {v['cohen_dz']:+.4f} | {v['paired_t_p']:.5g} | {v['holm_adjusted_p']:.5g} | {v['qm_like_wins']}/30 |")
    lines += ["", "Limitations:", ""] + ["- " + text for text in result["limitations"]]
    (run / "paired_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_rmse": comparisons}, indent=2))


if __name__ == "__main__":
    main()
