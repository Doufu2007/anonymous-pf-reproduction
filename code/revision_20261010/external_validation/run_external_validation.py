"""Run a frozen external synthetic-regime validation for the stable QGPF audit.

The script intentionally imports the stable implementation without modifying it.
All runs share generated truth/observations within a seed, while the filter has
separate deterministic RNG streams.  Truth is retained for evaluation only and
is never passed to ``run_filter``.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STABLE_DIR = REPO / "code" / "revision_20261009" / "qgpf"
if str(STABLE_DIR) not in sys.path:
    sys.path.insert(0, str(STABLE_DIR))
from qgpf_stable import FilterConfig, ModelConfig, run_filter, METHODS  # noqa: E402

METHODS = tuple(METHODS)
SEEDS = tuple(range(30))
PROTOCOL = HERE / "protocol.json"
F_EXT = np.array([[1.0, 0.8, 0.0, 0.0],
                  [0.0, 0.85, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.6],
                  [0.0, 0.0, 0.0, 0.90]], dtype=float)


def observation_ext(states: np.ndarray) -> np.ndarray:
    """Nonlinear range-like observation used only by this stress test."""
    p = np.asarray(states, dtype=np.float64)
    return np.sqrt(p[:, 0] ** 2 + 0.5 * p[:, 2] ** 2) + 0.2 * np.sin(0.1 * p[:, 0])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def git_provenance() -> dict[str, object]:
    try:
        def git(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True,
                                  text=True).stdout.strip()
        status = git("status", "--porcelain").splitlines()
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status),
                "status_repo_relative": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}


def generate_external(seed: int, model: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate one unseen trajectory with a different dynamics/observation regime."""
    rng = np.random.default_rng(10000 + seed)
    truth = np.zeros((model.horizon + 1, 4), dtype=float)
    observations = np.zeros(model.horizon + 1, dtype=float)
    truth[0] = np.asarray(model.initial_truth, dtype=float)
    process_std = np.asarray(model.process_std, dtype=float)
    for k in range(1, model.horizon + 1):
        truth[k] = truth[k - 1] @ F_EXT.T
        truth[k] += rng.normal(0.0, process_std)
        observations[k] = float(observation_ext(truth[k:k + 1])[0]
                                + rng.normal(0.0, model.observation_std))
    return truth, observations


def filter_seed(seed: int, method: str) -> int:
    return seed + {"B-stable": 2000003, "Q-stable": 3000006,
                   "Q-stable-retain": 4000009}[method]


def run_one(seed: int, method: str, out_dir: Path, model: ModelConfig,
            cfg: FilterConfig) -> dict[str, object]:
    run_dir = out_dir / method / f"seed-{seed:02d}"
    if run_dir.exists():
        raise FileExistsError(f"immutable run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = now()
    truth, observations = generate_external(seed, model)
    input_hashes = {"truth": hashlib.sha256(truth.tobytes()).hexdigest(),
                    "observations": hashlib.sha256(observations.tobytes()).hexdigest()}
    config = {"seed": seed, "method": method, "trajectory_seed": 10000 + seed,
              "filter_seed": filter_seed(seed, method), "model": model.__dict__,
              "filter": cfg.__dict__,
              "primary_metric": "pre-move position RMSE over k=1..T and coordinates x,y",
              "estimate_timing": "pre weighted-move estimate; post retained as diagnostic only",
              "truth_not_passed_to_filter": True, "protocol_sha256": sha(PROTOCOL),
              "stable_code_sha256": sha(STABLE_DIR / "qgpf_stable.py")}
    dump(run_dir / "config.json", config)
    dump(run_dir / "input_hashes.json", input_hashes)
    dump(run_dir / "environment.json", {"python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), "cpu": platform.processor(), "device": "CPU",
        "dtype": "float64", "threads": {k: os.environ.get(k) for k in
        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "git": git_provenance(), "timer_scope": "filter execution and in-memory diagnostics"})
    metrics: dict[str, object] = {}
    failure: dict[str, object] | None = None
    try:
        # The stable implementation is reused under a frozen, explicit
        # dynamics/observation substitution. This is a distribution-shift
        # stress test, not a recovery of the source author's model.
        import qgpf_stable as stable_module
        stable_module.F = F_EXT.copy()
        stable_module.observation = observation_ext
        rng = np.random.default_rng(filter_seed(seed, method))
        result = run_filter(observations, rng, method, model=model, cfg=cfg, record_post=True)
        pre = result["estimate_pre"]
        post = result["estimate_post"]
        err_pos = pre[1:, [0, 2]] - truth[1:, [0, 2]]
        err_vel = pre[1:, [1, 3]] - truth[1:, [1, 3]]
        metrics = {"seed": seed, "method": method,
                   "rmse_position": float(np.sqrt(np.mean(err_pos ** 2))),
                   "mae_position": float(np.mean(np.abs(err_pos))),
                   "rmse_velocity": float(np.sqrt(np.mean(err_vel ** 2))),
                   "mean_ess": float(np.mean(result["step_ess"])),
                   "mean_ess_over_n": float(np.mean(result["step_ess"]) / cfg.particle_count),
                   "mean_distinct": float(np.mean(result["step_distinct"])),
                   "runtime_s": float(result["runtime_s"]),
                   "initialization_runtime_s": float(result["initialization_runtime_s"]),
                   **{k: int(v) for k, v in result["counters"].items()},
                   "post_exploratory_rmse_position": float(np.sqrt(np.mean((post[1:, [0, 2]] - truth[1:, [0, 2]]) ** 2)))}
        arrays = {"truth": truth, "observations": observations,
                  "initial_particles": result["initial_particles"],
                  "estimate_pre": pre, "estimate_post": post,
                  "final_particles": result["final_particles"], "final_weights": result["final_weights"],
                  "particles_pre": result["particles_pre"], "particles_post": result["particles_post"],
                  "weights_pre": result["weights_pre"], "weights_post": result["weights_post"],
                  "step_counters": result["step_counters"], "step_ess": result["step_ess"],
                  "step_distinct": result["step_distinct"], "step_elapsed_s": result["step_elapsed_s"]}
        for name, arr in arrays.items():
            np.save(run_dir / f"{name}.npy", arr)
        dump(run_dir / "raw_metrics.json", metrics)
        dump(run_dir / "guardrails.json", result["guardrails"])
        dump(run_dir / "rng_final_state.json", result["rng_final_state"])
        with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", *[f"{tag}_{coord}" for tag in ("truth", "estimate_pre", "estimate_post")
                                      for coord in ("x", "vx", "y", "vy")]])
            for k in range(1, len(truth)):
                writer.writerow([k, *truth[k], *pre[k], *post[k]])
        status = "completed"
    except Exception as exc:  # keep failed artifacts, never silently drop a seed
        status = "failed"
        failure = {"exception_type": type(exc).__name__, "message": str(exc),
                   "traceback": traceback.format_exc(), "no_seed_exclusion": True}
        dump(run_dir / "failure.json", failure)
    (run_dir / "stdout.log").write_text(json.dumps({"status": status, "metrics": metrics}, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text(json.dumps(failure, ensure_ascii=False) + "\n" if failure else "", encoding="utf-8")
    files = {p.name: sha(p) for p in run_dir.iterdir() if p.is_file() and p.name != "manifest.json"}
    manifest = {"schema": "external-qgpf-validation.run.v1", "status": status,
                "run_id": f"external-{method}-seed-{seed:02d}", "started_at": started, "ended_at": now(),
                "method": method, "seed": seed, "trajectory_seed": 10000 + seed,
                "filter_seed": filter_seed(seed, method), "input_hashes": input_hashes,
                "protocol_sha256": sha(PROTOCOL), "stable_code_sha256": sha(STABLE_DIR / "qgpf_stable.py"),
                "files_sha256": files, "failure_artifact": "failure.json" if failure else None}
    dump(run_dir / "manifest.json", manifest)
    if status != "completed":
        raise RuntimeError(f"external run failed: {method} seed={seed}")
    return metrics


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 20000) -> tuple[float, float]:
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def write_summary(out_dir: Path, rows: list[dict[str, object]]) -> None:
    by_method = {method: {int(r["seed"]): r for r in rows if r["method"] == method} for method in METHODS}
    rng = np.random.default_rng(20261010)
    summary: dict[str, object] = {"schema": "external-qgpf-validation.summary.v1", "n_seeds": len(SEEDS),
                                  "unit": "trajectory seed", "primary_outcome": "rmse_position",
                                  "bootstrap_draws": 20000, "bootstrap_seed": 20261010, "comparisons": {}}
    for method in METHODS:
        metrics = [by_method[method][s] for s in SEEDS]
        summary[method] = {"rmse_position_mean": float(np.mean([r["rmse_position"] for r in metrics])),
                           "rmse_position_sd": float(np.std([r["rmse_position"] for r in metrics], ddof=1)),
                           "rmse_position_median": float(np.median([r["rmse_position"] for r in metrics])),
                           "rmse_position_iqr": [float(np.quantile([r["rmse_position"] for r in metrics], .25)),
                                                 float(np.quantile([r["rmse_position"] for r in metrics], .75))],
                           "velocity_rmse_mean": float(np.mean([r["rmse_velocity"] for r in metrics])),
                           "mean_ess_mean": float(np.mean([r["mean_ess"] for r in metrics])),
                           "mean_distinct_mean": float(np.mean([r["mean_distinct"] for r in metrics]))}
        if method != "B-stable":
            differences = np.array([by_method[method][s]["rmse_position"] - by_method["B-stable"][s]["rmse_position"] for s in SEEDS])
            ci = bootstrap_ci(differences, rng)
            summary["comparisons"][f"{method}_minus_B-stable"] = {
                "mean_difference": float(differences.mean()), "sd_difference": float(differences.std(ddof=1)),
                "median_difference": float(np.median(differences)), "bootstrap_95ci": list(ci),
                "dz": float(differences.mean() / differences.std(ddof=1)) if differences.std(ddof=1) else None,
                "wins_lower_rmse": int(np.sum(differences < 0)), "n": len(differences),
                "interpretation": "paired exploratory contrast; CI crossing zero is inconclusive, not equivalence"}
    dump(out_dir / "summary.json", summary)
    with (out_dir / "raw_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    out_dir = HERE / "runs-external-30"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"immutable output already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ModelConfig(horizon=100, initial_truth=(100.0, -2.0, -45.0, 5.0),
                        process_std=(2.0, 0.75, 1.25, 0.06), observation_std=2.0)
    cfg = FilterConfig(particle_count=200, initial_mean=(95.0, -1.5, -40.0, 4.5),
                       initial_std=(12.0, 2.0, 10.0, 2.0), bit_length=10, generation_count=4,
                       elite_fraction=0.20, rotation_angle_max_pi=0.08, rotation_angle_min_pi=0.01)
    rows: list[dict[str, object]] = []
    # Independent processes keep BLAS single-threaded and make the wall-clock
    # feasible without changing any scientific RNG stream.
    jobs = [(seed, method) for method in METHODS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one, seed, method, out_dir, model, cfg): (seed, method)
                   for seed, method in jobs}
        for future in as_completed(futures):
            seed, method = futures[future]
            rows.append(future.result())
            print(f"{now()} completed {method} seed={seed}", flush=True)
    rows.sort(key=lambda row: (METHODS.index(row["method"]), int(row["seed"])))
    write_summary(out_dir, rows)
    dump(out_dir / "batch_manifest.json", {"schema": "external-qgpf-validation.batch.v1", "status": "completed",
        "methods": METHODS, "seeds": SEEDS, "protocol_sha256": sha(PROTOCOL),
        "stable_code_sha256": sha(STABLE_DIR / "qgpf_stable.py"), "n_runs": len(rows),
        "summary_sha256": sha(out_dir / "summary.json"), "raw_metrics_csv_sha256": sha(out_dir / "raw_metrics.csv"),
        "git": git_provenance(), "external_input": "generated at run time; no archived trajectory files reused"})


if __name__ == "__main__":
    main()
