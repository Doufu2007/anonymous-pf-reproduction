"""Run the independent QM-HPF reproduction and preserve raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from qm_hpf import FilterConfig, ModelConfig, run_filter, simulate_truth


METHODS = ("bootstrap", "mh", "qm_hpf", "rotation_only", "crossover_only")


def parse_seeds(spec: str) -> list[int]:
    if ":" in spec:
        start, end = (int(x) for x in spec.split(":", 1))
        return list(range(start, end + 1))
    return [int(x) for x in spec.split(",") if x]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0:9")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/pilot"))
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    model = ModelConfig()
    cfg = FilterConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_code = Path(__file__).with_name("qm_hpf.py")
    manifest = {
        "schema": "qm_hpf.reproduction.v1",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "code_sha256": sha256_file(source_code),
        "model": asdict(model),
        "filter": asdict(cfg),
        "seeds": seeds,
        "methods": METHODS,
        "assumptions": json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8")),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows: list[dict[str, object]] = []
    for seed in seeds:
        truth, observations = simulate_truth(np.random.default_rng(seed), model)
        for method in METHODS:
            run_rng = np.random.default_rng(seed + 1000003 * (METHODS.index(method) + 1))
            result = run_filter(observations, truth, run_rng, method, model, cfg)
            np.save(args.out_dir / f"truth_seed{seed}.npy", truth)
            np.save(args.out_dir / f"observations_seed{seed}.npy", observations)
            np.save(args.out_dir / f"estimate_{method}_seed{seed}.npy", result["estimate"])
            rows.append({"seed": seed, "method": method, **{k: result[k] for k in ("rmse", "mae", "mean_ess", "mean_distinct", "runtime_s", "likelihood_evals", "candidate_evals", "old_evals", "proposal_attempts", "accepts")}})
    with (args.out_dir / "raw_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, dict[str, float | int]] = {}
    for method in METHODS:
        values = [r for r in rows if r["method"] == method]
        summary[method] = {"runs": len(values)}
        for metric in ("rmse", "mae", "mean_ess", "mean_distinct", "runtime_s", "likelihood_evals", "candidate_evals", "old_evals", "proposal_attempts", "accepts"):
            arr = np.array([float(v[metric]) for v in values])
            summary[method][metric + "_mean"] = float(np.mean(arr))
            summary[method][metric + "_sd"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
