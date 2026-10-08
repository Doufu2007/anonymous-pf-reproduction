"""Run the frozen exploratory fair-budget supplement and retain raw bundles."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from contextlib import redirect_stdout

import numpy as np
import scipy

from budget_benchmark import METHODS, run_budget_filter
from qm_hpf import FilterConfig, ModelConfig, simulate_truth


HERE = Path(__file__).resolve().parent
# The publication copy is self-contained.  The source papers and extraction
# files used to establish provenance are deliberately not redistributed.
ROOT = HERE.parents[1]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def environment():
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    numpy_config = io.StringIO()
    with redirect_stdout(numpy_config):
        np.show_config()
    return {"python": sys.version, "python_executable": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "cpu": cpu, "logical_cpu_count": os.cpu_count(),
            "device": "CPU", "dtype": "float64", "AMP": "not applicable",
            "thread_environment": {key: os.environ.get(key, "not explicitly set") for key in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")},
            "numpy_config": numpy_config.getvalue(),
            "randomness_contract": "Explicit NumPy default_rng generators; no global RNG or model training. Same-environment reproducibility only."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=HERE / "fair_budget_plan.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "runs/fair-budget-30")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite a run bundle: {args.out_dir}")
    if plan["methods"] != list(METHODS) or plan["seeds"] != list(range(30)):
        raise ValueError("frozen matrix identity mismatch")
    args.out_dir.mkdir(parents=True)
    out = args.out_dir
    start = now()
    env = environment()
    dump(out / "environment.json", env)
    dump(out / "frozen_plan.json", plan)
    code_names = ("budget_benchmark.py", "run_budget_benchmark.py", "test_budget_benchmark.py", "analyze_budget_benchmark.py", "qm_hpf.py")
    code_hashes = {name: sha(HERE / name) for name in code_names}
    source_provenance = HERE / "source_provenance.json"
    source_hashes = json.loads(source_provenance.read_text(encoding="utf-8"))
    git = subprocess.run(["git", "status", "--porcelain=v1"], cwd=ROOT, capture_output=True, text=True)
    git_status = {"exit_code": git.returncode, "stdout": git.stdout, "stderr": git.stderr}
    test = subprocess.run([sys.executable, "-m", "unittest", "test_budget_benchmark", "-v"],
                           cwd=HERE, capture_output=True, text=True, encoding="utf-8",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    (out / "test_stdout.log").write_text(test.stdout, encoding="utf-8")
    (out / "test_stderr.log").write_text(test.stderr, encoding="utf-8")
    dump(out / "test_evidence.json", {"command": [sys.executable, "-m", "unittest", "test_budget_benchmark", "-v"],
            "exit_code": test.returncode, "test_count": 6, "mathematical_tests":
            ["full transition plus likelihood MH density ratio and reverse detailed balance",
             "analytic linear-observation conditional Gaussian agreement",
             "joint ancestor/state/cached-likelihood resampling",
             "low-set generation cap without forced budget expenditure",
             "likelihood cap and normalization", "same-seed same-environment deterministic values excluding timing"]})
    if test.returncode:
        dump(out / "failure.json", {"status": "failed", "reason": "math/contract tests failed", "time": now()})
        raise RuntimeError(test.stderr)
    manifest = {"schema": "particle-filter.budget-audit.v1", "status": "running", "started_at": start,
                "analysis_mode": "exploratory supplemental comparison", "argv": sys.argv,
                "plan_sha256": sha(args.plan), "code_sha256": code_hashes,
                "source_sha256": source_hashes, "git_status": git_status,
                "environment_sha256": sha(out / "environment.json"), "model": asdict(ModelConfig()),
                "filter_defaults": asdict(FilterConfig()), "seeds": plan["seeds"], "methods": METHODS,
                "likelihood_cap_per_step": 200, "likelihood_cap_per_run": 40000,
                "allowance": "local CPU only; user-authorized supplementary fair comparison; no tuning or seed exclusions"}
    dump(out / "manifest.json", manifest)
    rows = []
    log_lines = [f"started {start}", "Tests passed: 6 / 6", "This is exploratory; previously observed native-cost results are disclosed in the frozen plan."]
    model = ModelConfig()
    for seed in plan["seeds"]:
        truth, obs = simulate_truth(np.random.default_rng(seed), model, process="gaussian", process_variance=1.0)
        for method_index, method in enumerate(METHODS):
            run_dir = out / "runs" / f"seed-{seed:02d}" / method
            run_dir.mkdir(parents=True)
            run_started = now()
            stream_seed = seed + 1000003 * (method_index + 1)
            config = {"method": method, "seed": seed, "truth_rng_seed": seed, "algorithm_rng_seed": stream_seed,
                      "model": asdict(model), "process_variance": 1.0, "process_family": "Gaussian",
                      "particle_count": 200 if method == "bootstrap_200" else 100,
                      "initial_particles": "Normal(1,1 variance)", "proposal_std": 0.1 if method == "resample_move_100" else None,
                      "filter": asdict(FilterConfig()), "pre_move_estimate": True,
                      "qm_round_robin_candidate_cap": 100, "qm_max_generations_per_low_particle": 20,
                      "counting": "scalar likelihood evaluations; cached old states; no dummy calls",
                      "native_like_kernel": "heuristic: likelihood-only acceptance and weight-ratio update, not posterior invariant"}
            dump(run_dir / "config.json", config)
            try:
                result = run_budget_filter(obs, truth, np.random.default_rng(stream_seed), method, model,
                                           proposal_std=plan["resample_move_proposal_std"])
                for name, arr in [("truth", truth), ("observations", obs), ("estimate", result["estimate"]),
                                  ("final_particles", result["final_particles"]),
                                  ("step_likelihood_evals", result["step_likelihood_evals"]),
                                  ("step_proposal_attempts", result["step_proposal_attempts"])]:
                    np.save(run_dir / (name + ".npy"), arr)
                metrics = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
                dump(run_dir / "raw_metrics.json", metrics)
                guardrails = {"finite_metrics": bool(all(np.isfinite(v) for v in metrics.values())),
                              "budget_cap_pass": bool(result["likelihood_evals"] <= 40000),
                              "step_cap_pass": bool(np.all(result["step_likelihood_evals"] <= 200)),
                              "equal_full_expenditure": result["likelihood_evals"] == 40000,
                              "normalized_weights": bool(np.isclose(result["final_weight_sum"], 1.0)),
                              "interpretation": "valid cap; report shortfalls without dropping runs or forcing candidates"}
                dump(run_dir / "guardrails.json", guardrails)
                (run_dir / "stdout.log").write_text(f"completed {method} seed={seed}; likelihood_evals={result['likelihood_evals']}; cap=40000\n", encoding="utf-8")
                (run_dir / "stderr.log").write_text("", encoding="utf-8")
                files = {p.name: sha(p) for p in run_dir.iterdir() if p.is_file()}
                dump(run_dir / "manifest.json", {"schema": "particle-filter.budget-audit.run.v1", "status": "completed",
                    "run_id": f"{method}-seed-{seed:02d}", "started_at": run_started, "ended_at": now(),
                    "code_sha256": code_hashes, "frozen_plan_sha256": sha(args.plan),
                    "environment_sha256": manifest["environment_sha256"], "files_sha256": files})
                rows.append({"seed": seed, "method": method, **metrics})
                log_lines.append(f"completed {method} seed={seed}; likelihood_evals={result['likelihood_evals']}")
            except Exception:
                failure = traceback.format_exc()
                (run_dir / "stderr.log").write_text(failure, encoding="utf-8")
                dump(run_dir / "failure.json", {"status": "failed", "time": now(), "reason": failure})
                manifest.update(status="failed", ended_at=now(), failed_run=str(run_dir))
                dump(out / "manifest.json", manifest)
                raise
    with (out / "raw_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # One additional, independent same-environment replay for seed 0, all
    # methods. No timing or ESS/second equality is required or claimed.
    replay_dir = out / "fixed-repro-seed-00"
    replay_dir.mkdir()
    truth, obs = simulate_truth(np.random.default_rng(0), model)
    replay_checks = []
    for mi, method in enumerate(METHODS):
        replay = run_budget_filter(obs, truth, np.random.default_rng(1000003 * (mi + 1)), method, model)
        target = out / "runs/seed-00" / method
        np.save(replay_dir / (method + "_estimate.npy"), replay["estimate"])
        values = {k: v for k, v in replay.items() if not isinstance(v, np.ndarray) and k not in {"runtime_s", "ess_per_second"}}
        dump(replay_dir / (method + "_raw_metrics.json"), values)
        original = json.loads((target / "raw_metrics.json").read_text(encoding="utf-8"))
        original = {k: v for k, v in original.items() if k not in {"runtime_s", "ess_per_second"}}
        same = bool(np.array_equal(replay["estimate"], np.load(target / "estimate.npy")) and values == original)
        replay_checks.append({"method": method, "estimate_and_non_timing_metrics_identical": same,
                              "reference_run": str(target.relative_to(out)),
                              "replay_estimate_sha256": sha(replay_dir / (method + "_estimate.npy")),
                              "reference_estimate_sha256": sha(target / "estimate.npy")})
    dump(replay_dir / "verification.json", {"same_environment_only": True, "checks": replay_checks})
    if not all(item["estimate_and_non_timing_metrics_identical"] for item in replay_checks):
        raise RuntimeError("same-environment fixed-seed replay failed")
    manifest.update(status="completed", ended_at=now(), completed_runs=len(rows), excluded_runs=0,
                    raw_metrics_sha256=sha(out / "raw_metrics.csv"),
                    fixed_repro_evidence_sha256=sha(replay_dir / "verification.json"))
    dump(out / "manifest.json", manifest)
    log_lines.append(f"ended {manifest['ended_at']}; completed=90; excluded=0; fixed-seed replay passed")
    (out / "stdout.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (out / "stderr.log").write_text("", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "completed_runs": len(rows), "status": "completed",
                      "test_count": 6, "fixed_repro_pass": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
