"""Run a small predeclared sensitivity sweep for omitted QM-HPF settings."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from qm_hpf import FilterConfig, ModelConfig, run_filter, simulate_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/sensitivity"))
    parser.add_argument("--seeds", default="0:4")
    args = parser.parse_args()
    start_seed, end_seed = (int(x) for x in args.seeds.split(":", 1))
    seeds = list(range(start_seed, end_seed + 1))
    generations = (5, 20, 40)
    bit_lengths = (8, 16)
    ranges = ((-5.0, 5.0), (-10.0, 10.0))
    model = ModelConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for generation in generations:
        for bit_length in bit_lengths:
            for state_min, state_max in ranges:
                cfg = replace(FilterConfig(), generation_count=generation, bit_length=bit_length,
                              state_min=state_min, state_max=state_max)
                setting = f"G{generation}_B{bit_length}_R{state_min:g}_{state_max:g}"
                for seed in seeds:
                    truth, observations = simulate_truth(np.random.default_rng(seed), model)
                    for method_index, method in enumerate(("bootstrap", "qm_hpf")):
                        rng = np.random.default_rng(seed + 2000003 * (method_index + 1) + generation * 97 + bit_length * 13)
                        result = run_filter(observations, truth, rng, method, model, cfg)
                        rows.append({
                            "setting": setting, "generation_count": generation,
                            "bit_length": bit_length, "state_min": state_min,
                            "state_max": state_max, "seed": seed, "method": method,
                            "rmse": result["rmse"], "mae": result["mae"],
                            "mean_ess": result["mean_ess"], "mean_distinct": result["mean_distinct"],
                            "runtime_s": result["runtime_s"],
                        })
    with (args.out_dir / "raw_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate: dict[str, dict[str, float | int]] = {}
    for setting in sorted({str(row["setting"]) for row in rows}):
        values = [row for row in rows if row["setting"] == setting]
        aggregate[setting] = {}
        for method in ("bootstrap", "qm_hpf"):
            subset = [row for row in values if row["method"] == method]
            rmse = np.array([float(row["rmse"]) for row in subset])
            aggregate[setting][f"{method}_runs"] = len(subset)
            aggregate[setting][f"{method}_rmse_mean"] = float(np.mean(rmse))
            aggregate[setting][f"{method}_rmse_sd"] = float(np.std(rmse, ddof=1))
            aggregate[setting][f"{method}_runtime_mean"] = float(np.mean([float(row["runtime_s"]) for row in subset]))
        aggregate[setting]["qm_minus_bootstrap_rmse"] = aggregate[setting]["qm_hpf_rmse_mean"] - aggregate[setting]["bootstrap_rmse_mean"]
    manifest = {"schema": "qm_hpf.sensitivity.v1", "seeds": seeds,
                "generations": generations, "bit_lengths": bit_lengths,
                "state_ranges": ranges, "model": asdict(model),
                "interpretation": "Exploratory sensitivity to source-omitted implementation settings; not a replacement for the main predeclared run."}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
