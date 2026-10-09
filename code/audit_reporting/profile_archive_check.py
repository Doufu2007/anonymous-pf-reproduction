"""Check profiling replay outputs against existing scientific result archives."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    args = parser.parse_args()
    profile_dir = args.profile_dir.resolve()
    checks = []
    qg_rows = list(csv.DictReader((REPO / "code/qgpf_reproduction/runs-main-30/raw_metrics.csv").open(encoding="utf-8")))
    for method in ("bootstrap_200", "resample_move_100", "qm_hpf_like_100", "qgpf_like", "ecqpf_source_module"):
        for seed in (0, 1, 2):
            run = profile_dir / method / f"seed-{seed:02d}"
            output = json.loads((run / "uninstrumented_outputs.json").read_text(encoding="utf-8"))
            artifacts = []
            def check_array(name, array, reference):
                expected = np.load(reference)
                actual = np.asarray(array)
                shape_matches = actual.shape == expected.shape
                bitwise = bool(shape_matches and np.array_equal(actual, expected))
                close = bool(shape_matches and np.allclose(actual, expected, rtol=1e-12, atol=1e-12))
                artifacts.append({"output": name, "archive": reference.relative_to(REPO).as_posix(),
                                  "archive_sha256": sha(reference), "shape_matches": shape_matches,
                                  "bitwise_equal": bitwise, "within_tolerance": close,
                                  "maximum_absolute_difference": float(np.max(np.abs(actual - expected))) if shape_matches else None})
            if method in ("bootstrap_200", "resample_move_100", "qm_hpf_like_100"):
                archive = REPO / "code/qm_hpf_reproduction/runs/fair-budget-30/runs" / f"seed-{seed:02d}" / method
                for name in ("estimate", "final_particles", "step_likelihood_evals", "step_proposal_attempts"):
                    check_array(name, output[name]["values"], archive / f"{name}.npy")
                metrics = json.loads((archive / "raw_metrics.json").read_text(encoding="utf-8"))
                metric_values = {k: v for k, v in metrics.items() if k not in {"runtime_s", "ess_per_second"}}
                equal_metrics = all(output[k] == v for k, v in metric_values.items())
                artifacts.append({"output": "scientific_metrics", "archive": (archive / "raw_metrics.json").relative_to(REPO).as_posix(),
                                  "archive_sha256": sha(archive / "raw_metrics.json"), "bitwise_equal": equal_metrics,
                                  "within_tolerance": all(np.isclose(output[k], v, rtol=1e-12, atol=1e-12) for k, v in metric_values.items())})
            elif method == "qgpf_like":
                archive = REPO / "code/qgpf_reproduction/runs-main-30"
                check_array("estimate", output["estimate"]["values"], archive / f"estimate_qgpf_seed{seed}.npy")
                metric_values = next(r for r in qg_rows if int(r["seed"]) == seed and r["method"] == "qgpf")
                names = [k for k in metric_values if k not in {"seed", "method", "runtime_s"}]
                artifacts.append({"output": "scientific_metrics", "archive": (archive / "raw_metrics.csv").relative_to(REPO).as_posix(),
                                  "archive_sha256": sha(archive / "raw_metrics.csv"),
                                  "bitwise_equal": all(float(output[k]) == float(metric_values[k]) for k in names),
                                  "within_tolerance": all(np.isclose(output[k], float(metric_values[k]), rtol=1e-12, atol=1e-12) for k in names)})
            else:
                archive = REPO / "code/source_aligned_audit/runs-main-30" / f"seed-{seed:02d}"
                check_array("source_estimates", [row["estimate"]["values"] for row in output], archive / "source_ecqpf_operator_estimates.npy")
                check_array("source_indices", [row["indices"]["values"] for row in output], archive / "source_ecqpf_operator_indices.npy")
                raw_rows = [r for r in [json.loads(line) for line in (archive / "raw_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
                            if r["method"] == "source_ecqpf_operator"]
                metric_keys = ("fitness_evaluations", "candidate_distance_evaluations", "binary_dimensions_processed",
                               "likelihood_evaluations", "generations", "empty_subsets", "distinct_count", "ellipse_count",
                               "logfitness", "optimal_logfitness", "fitness_variance")
                artifacts.append({"output": "source_scientific_metrics", "archive": (archive / "raw_metrics.jsonl").relative_to(REPO).as_posix(),
                                  "archive_sha256": sha(archive / "raw_metrics.jsonl"),
                                  "bitwise_equal": all(output[i][k] == raw_rows[i][k] for i in range(20) for k in metric_keys),
                                  "within_tolerance": all(np.isclose(output[i][k], raw_rows[i][k], rtol=1e-12, atol=1e-12) for i in range(20) for k in metric_keys)})
            checks.append({"method": method, "seed": seed, "passed": all(a["within_tolerance"] for a in artifacts),
                           "all_bitwise_equal": all(a["bitwise_equal"] for a in artifacts), "checks": artifacts})
    report = {"schema": "particle-filter.profile-archive-replay-check.v1", "status": "PASS" if all(r["passed"] for r in checks) else "FAIL",
              "workloads": len(checks), "all_bitwise_equal": all(r["all_bitwise_equal"] for r in checks),
              "tolerance": {"rtol": 1e-12, "atol": 1e-12}, "timing_fields_excluded": ["runtime_s", "ess_per_second"], "checks": checks}
    (profile_dir / "profile_archive_replay_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "workloads", "all_bitwise_equal")}))
    if report["status"] != "PASS":
        raise AssertionError("Archive replay check failed; preserve outputs and investigate")


if __name__ == "__main__":
    main()
