"""Fail-closed entry point for a source-faithful QM-HPF reproduction.

The numerical kernel is intentionally gated by the source contract.  The
existing parent implementation remains the separately labelled QM-HPF-like
audit.  This runner cannot silently inherit its completion choices.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_strict_common import MissingSourceEvidence, load_json, manifest_base, require_resolved, write_failure
from source_kernel import OneDConfig, run_filter, simulate_truth


REQUIRED_ONE_D_FIELDS = (
    "initial_particle_distribution",
    "initial_particle_std",
    "quantum_encoding_range",
    "bit_length",
    "continuous_state_decode_rule",
    "state_min",
    "state_max",
    "generation_count",
    "amplitude_persistence_across_generations",
    "amplitude_mapping",
    "low_weight_particle_iteration_order",
    "random_number_consumption_order",
    "recompute_all_weights_after_acceptance",
    "final_systematic_resampling",
    "non_gaussian_process_noise",
    "rotation_sign_table",
    "rotation_angle_schedule",
    "execution_mode",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("source_config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("runs-strict"))
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        if config.get("task") != "one_dimensional":
            raise MissingSourceEvidence("only the source one-dimensional task is enabled in this stage")
        require_resolved(config, REQUIRED_ONE_D_FIELDS)
        if config.get("execution_mode") != "source_strict_after_evidence":
            raise MissingSourceEvidence("set execution_mode=source_strict_after_evidence only after the source evidence is frozen")
        signs = config.get("rotation_sign_table")
        if not isinstance(signs, dict):
            raise MissingSourceEvidence("rotation_sign_table must be a resolved JSON object")
        kernel_cfg = OneDConfig(
            horizon=int(config["horizon"]),
            particle_count=int(config["particle_count"]),
            initial_state=float(config["initial_state"]),
            observation_variance=float(config["observation_variance"]),
            initial_particle_std=float(config["initial_particle_std"]),
            state_min=float(config["state_min"]),
            state_max=float(config["state_max"]),
            bit_length=int(config["bit_length"]),
            generation_count=int(config["generation_count"]),
            amplitude_persistence=bool(config["amplitude_persistence_across_generations"]),
            final_systematic_resampling=bool(config["final_systematic_resampling"]),
            rotation_angle_max_pi=0.04,
            rotation_angle_min_pi=0.01,
            low_order=str(config["low_weight_particle_iteration_order"]),
            amplitude_mapping=str(config["amplitude_mapping"]),
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        seeds = [int(x) for x in config.get("seeds", [0])]
        rows = []
        for seed in seeds:
            truth_rng = np.random.default_rng(seed)
            filter_rng = np.random.default_rng(seed + 1000003)
            process_variance = float(config["process_variances"][0])
            truth, observations = simulate_truth(truth_rng, kernel_cfg, process_variance)
            result = run_filter(observations, filter_rng, kernel_cfg, process_variance, {str(k): float(v) for k, v in signs.items()})
            np.savez_compressed(args.out_dir / f"raw_seed{seed}.npz", truth=truth, observations=observations, estimate=result["estimate"], final_particles=result["final_particles"], final_weights=result["final_weights"])
            with (args.out_dir / f"trace_seed{seed}.pkl").open("wb") as handle:
                pickle.dump(result["traces"], handle, protocol=5)
            error = result["estimate"][1:] - truth[1:]
            rows.append({"seed": seed, "process_variance": process_variance, "rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(np.abs(error)))})
        (args.out_dir / "raw_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        manifest = manifest_base(run_id=f"QM-HPF-SOURCE-STRICT__{args.out_dir.name}", method="QM-HPF", config_path=args.config, config=config)
        manifest.update({"status": "COMPLETED", "source_faithful_claim_allowed": True, "outputs": [str(p.name) for p in args.out_dir.iterdir() if p.is_file()]})
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"status": "COMPLETED", "method": "QM-HPF", "runs": len(rows)}, ensure_ascii=False))
        return 0
    except (MissingSourceEvidence, ValueError, json.JSONDecodeError) as exc:
        failure = args.out_dir / "failure_artifacts" / "strict_blocked.json"
        write_failure(failure, method="QM-HPF", message=str(exc), config_path=args.config)
        print(json.dumps({"status": "BLOCKED", "method": "QM-HPF", "failure": str(failure), "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
