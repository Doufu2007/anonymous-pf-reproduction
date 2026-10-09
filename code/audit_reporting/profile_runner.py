"""Non-invasive, fixed-seed CPU profiling supplement for existing audit code.

Run from the repository root with the environment in profile_README.md.
Scientific source functions are imported unchanged. Time is partitioned by
function-exclusive cProfile time, not estimated phase-inclusive time.
"""
from __future__ import annotations

import argparse
import cProfile
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CODE = REPO / "code"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
if os.environ.get("PYTHONHASHSEED") != "0" or any(os.environ.get(k) != "1" for k in THREAD_KEYS):
    raise RuntimeError("Set PYTHONHASHSEED=0 and every thread environment variable listed in profile_README.md to 1 before startup.")

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

sys.path[:0] = [str(CODE / name) for name in
               ("qm_hpf_reproduction", "qgpf_reproduction", "source_aligned_audit")]
import budget_benchmark as budget
import qm_hpf as qm
import qgpf as qg
import ecqpf_operator as ec

BUCKETS = ("likelihood", "rotation_crossover_encoding", "measurement",
           "probability_fitness", "rng", "sorting_resampling", "other")
PLAN = HERE / "profile_frozen_plan.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative_filename(filename):
    """Remove personal paths from both profile keys and caller keys."""
    if filename.startswith("~") or filename.startswith("<"):
        return filename
    path = Path(filename)
    if not path.is_absolute():
        return filename.replace("\\", "/")
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        parts = list(path.parts)
        for marker in ("site-packages", "Lib", "lib"):
            if marker in parts:
                return "<runtime>/" + "/".join(parts[parts.index(marker):])
        return "<runtime>/" + path.name


def function_key(key):
    filename, line, name = key
    return (relative_filename(filename), line, name)


def category(key):
    filename, _line, name = key
    # Only named, interpretable functions are attributed. Shared NumPy helpers
    # and the self time of mixed-purpose loops remain in the residual bucket.
    if filename.startswith("code/"):
        if name in {"particle_logweights", "likelihood", "observation"}:
            return "likelihood"
        if name in {"quantum_rotation_candidate", "crossover_candidate", "encode_scalar",
                    "decode_scalar", "rotate_bit_toward", "encode_state", "decode_state",
                    "_rotate_probability"}:
            return "rotation_crossover_encoding"
        if name == "observe_binary":
            return "measurement"
        if name in {"normalize_logweights", "conditional_log_ratio", "ess",
                    "subset_logfitness", "update_swarm", "exact_fitness_subset"}:
            return "probability_fitness"
        if name in {"systematic_resample", "resample_pairs", "duplicate_indices"}:
            return "sorting_resampling"
    if "numpy.random" in name or "_generator.Generator" in name:
        return "rng"
    # Named sorting primitives have a unique, interpretable operation, even
    # when invoked through common NumPy wrappers. Other reductions stay other.
    if any(word in name for word in ("'argsort'", "'sort'", "'searchsorted'")):
        return "sorting_resampling"
    return "other"


def scientific(value):
    """Canonical deterministic outputs; timing-derived values are excluded."""
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite scientific output")
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items()
                if k not in {"runtime_s", "ess_per_second"}}
    if isinstance(value, (list, tuple)):
        return [scientific(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def prepare(method, seed):
    """Recreate the original inputs and RNG streams, outside the timed call."""
    if method in budget.METHODS:
        model = qm.ModelConfig()
        truth, obs = qm.simulate_truth(np.random.default_rng(seed), model,
                                      process="gaussian", process_variance=1.0)
        mi = budget.METHODS.index(method)
        algorithm_seed = seed + 1000003 * (mi + 1)
        return (lambda: budget.run_budget_filter(obs, truth,
                np.random.default_rng(algorithm_seed), method, model)), {
            "method": method, "seed": seed, "algorithm_rng_seed": algorithm_seed,
            "model": asdict(model), "filter_defaults": asdict(qm.FilterConfig()),
            "inputs_sha256": canonical_hash(scientific({"truth": truth, "observations": obs}))}
    if method == "qgpf_like":
        model, cfg = qg.ModelConfig(), qg.FilterConfig()
        truth, obs = qg.simulate_truth(np.random.default_rng(seed), model)
        # Original METHODS=(bootstrap,qgpf); qgpf therefore has stream index 2.
        algorithm_seed = seed + 1000003 * 2
        return (lambda: qg.run_filter(obs, truth,
                np.random.default_rng(algorithm_seed), "qgpf", model, cfg)), {
            "method": method, "seed": seed, "algorithm_rng_seed": algorithm_seed,
            "model": asdict(model), "filter": asdict(cfg),
            "inputs_sha256": canonical_hash(scientific({"truth": truth, "observations": obs}))}
    if method == "ecqpf_source_module":
        original_plan = json.loads((CODE / "source_aligned_audit/frozen_plan.json").read_text(encoding="utf-8"))
        cfg = ec.OperatorConfig()
        streams = np.random.SeedSequence(seed).spawn(7)
        truth_rng, obs_rng, particle_rng = [np.random.default_rng(s) for s in streams[:3]]
        updates = original_plan["updates_per_seed"]
        truth = truth_rng.normal(0, np.sqrt(original_plan["generator"]["prior_variance"]), (updates, 2))
        centers = truth + obs_rng.normal(0, np.sqrt(original_plan["generator"]["observation_variance"]), (updates, 2))
        clouds = particle_rng.normal(0, np.sqrt(original_plan["generator"]["prior_variance"]),
                                     (updates, cfg.particle_count, 2))
        def call():
            rng = np.random.default_rng(streams[3])
            return [ec.source_select(clouds[i], centers[i], cfg, rng) for i in range(updates)]
        return call, {"method": method, "seed": seed, "operator": asdict(cfg),
                      "updates": updates, "algorithm_rng": "SeedSequence(seed).spawn(7)[3]",
                      "inputs_sha256": canonical_hash(scientific({"centers": centers, "clouds": clouds})),
                      "truth_not_passed_to_operator": True,
                      "source_plan_sha256": sha(CODE / "source_aligned_audit/frozen_plan.json")}
    raise ValueError(method)


def environment():
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    pools = [{k: v for k, v in info.items() if k != "filepath"} for info in threadpool_info()]
    if any(info["num_threads"] != 1 for info in pools):
        raise RuntimeError("Runtime threadpool is not single-threaded")
    return {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "cpu": cpu, "logical_cpu_count": os.cpu_count(), "device": "CPU", "dtype": "float64",
            "python_hash_seed": os.environ["PYTHONHASHSEED"],
            "thread_environment": {k: os.environ[k] for k in THREAD_KEYS}, "threadpools": pools,
            "executable": Path(sys.executable).name, "gpu": "not used",
            "scope": "same machine and environment; no CPU affinity/isolation guarantee"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError("Refusing to overwrite profiling bundle")
    out.mkdir(parents=True)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan["seeds"] != [0, 1, 2] or plan["methods"] != list(budget.METHODS) + ["qgpf_like", "ecqpf_source_module"]:
        raise ValueError("Frozen profiling matrix identity mismatch")
    sources = [CODE / "qm_hpf_reproduction/qm_hpf.py", CODE / "qm_hpf_reproduction/budget_benchmark.py",
               CODE / "qgpf_reproduction/qgpf.py", CODE / "source_aligned_audit/ecqpf_operator.py",
               Path(__file__).resolve()]
    code_hashes = {path.relative_to(REPO).as_posix(): sha(path) for path in sources}
    for name, expected in plan["source_code_sha256"].items():
        if code_hashes[name] != expected:
            raise RuntimeError("Frozen algorithm source changed")
    started = datetime.now(timezone.utc).isoformat()
    dump(out / "profile_frozen_plan.json", plan)
    rows = []
    with threadpool_limits(limits=1):
        dump(out / "profile_environment.json", environment())
        for method in plan["methods"]:
            for seed in plan["seeds"]:
                run_dir = out / method / f"seed-{seed:02d}"
                run_dir.mkdir(parents=True)
                baseline_call, config = prepare(method, seed)
                dump(run_dir / "config.json", config)
                begin = time.perf_counter()
                baseline_raw = baseline_call()
                baseline_wall = time.perf_counter() - begin
                baseline = scientific(baseline_raw)
                dump(run_dir / "uninstrumented_outputs.json", baseline)
                profiled_call, repeated_config = prepare(method, seed)
                if repeated_config != config:
                    raise AssertionError("Input/config identity mismatch")
                profiler = cProfile.Profile()
                begin = time.perf_counter()
                instrumented_raw = profiler.runcall(profiled_call)
                instrumented_wall = time.perf_counter() - begin
                instrumented = scientific(instrumented_raw)
                dump(run_dir / "instrumented_outputs.json", instrumented)
                equal = canonical_hash(baseline) == canonical_hash(instrumented)
                if not equal:
                    dump(run_dir / "failure.json", {"status": "failed", "reason": "profiling changed scientific output"})
                    raise AssertionError("Profiling changed scientific output")
                stats = pstats.Stats(profiler)
                stats.stats = {function_key(key): (cc, nc, tt, ct,
                              {function_key(caller): value for caller, value in callers.items()})
                              for key, (cc, nc, tt, ct, callers) in stats.stats.items()}
                stats.dump_stats(str(run_dir / "exclusive_profile.prof"))
                totals = {bucket: 0.0 for bucket in BUCKETS}
                function_rows = []
                for key, (cc, nc, tt, ct, callers) in stats.stats.items():
                    bucket = category(key)
                    totals[bucket] += tt
                    function_rows.append({"file": key[0], "line": key[1], "function": key[2],
                                          "primitive_calls": cc, "total_calls": nc,
                                          "exclusive_time_s": tt, "cumulative_time_s": ct, "bucket": bucket})
                exclusive_total = sum(totals.values())
                percents = {k: v / exclusive_total * 100 for k, v in totals.items()}
                if not np.isclose(sum(percents.values()), 100, rtol=0, atol=1e-10):
                    raise AssertionError("Nonexclusive timing partition")
                function_rows.sort(key=lambda item: -item["exclusive_time_s"])
                for function_row in function_rows:
                    function_row["exclusive_total_share_percent"] = function_row["exclusive_time_s"] / exclusive_total * 100
                dump(run_dir / "exclusive_functions.json", function_rows)
                dump(run_dir / "top_functions.json", function_rows[:20])
                result = {"method": method, "seed": seed, "status": "completed",
                          "scientific_outputs_identical": equal, "scientific_output_sha256": canonical_hash(baseline),
                          "uninstrumented_wall_s": baseline_wall, "instrumented_wall_s": instrumented_wall,
                          "profiler_overhead_s": instrumented_wall - baseline_wall,
                          "profiler_overhead_percent": (instrumented_wall / baseline_wall - 1) * 100,
                          "cprofile_exclusive_total_s": exclusive_total,
                          "bucket_exclusive_time_s": totals, "bucket_exclusive_percent": percents,
                          "measurement_separable": method == "ecqpf_source_module",
                          "measurement_unavailable_reason": None if method == "ecqpf_source_module" else
                              "No separate measurement function; none for classical baselines, embedded in rotation/loop for QM/QGPF.",
                          "plan_sha256": sha(PLAN), "source_code_sha256": code_hashes}
                dump(run_dir / "profile_result.json", result)
                rows.append(result)
                print(json.dumps({"method": method, "seed": seed, "scientific_outputs_identical": equal,
                                  "uninstrumented_wall_s": baseline_wall}), flush=True)
    summary = []
    for method in plan["methods"]:
        records = [r for r in rows if r["method"] == method]
        item = {"method": method, "runs": len(records), "all_scientific_outputs_identical": all(r["scientific_outputs_identical"] for r in records),
                "uninstrumented_wall_s_mean": float(np.mean([r["uninstrumented_wall_s"] for r in records])),
                "instrumented_wall_s_mean": float(np.mean([r["instrumented_wall_s"] for r in records])),
                "profiler_overhead_percent_mean": float(np.mean([r["profiler_overhead_percent"] for r in records])),
                "bucket_exclusive_percent_mean": {k: float(np.mean([r["bucket_exclusive_percent"][k] for r in records])) for k in BUCKETS},
                "measurement_independent_time": "available" if method == "ecqpf_source_module" else
                    "not_applicable" if method in {"bootstrap_200", "resample_move_100"} else "unavailable",
                "joint_rotate_measure_function": "rotate_bit_toward" if method == "qm_hpf_like_100" else None}
        aggregate = {}
        for seed in plan["seeds"]:
            function_rows = json.loads((out / method / f"seed-{seed:02d}/exclusive_functions.json").read_text(encoding="utf-8"))
            for row in function_rows:
                key = (row["file"], row["line"], row["function"], row["bucket"])
                aggregate.setdefault(key, []).append(row["exclusive_total_share_percent"])
        item["top_functions_mean_exclusive_share"] = sorted([
            {"file": key[0], "line": key[1], "function": key[2], "bucket": key[3],
             "exclusive_total_share_percent_mean": sum(values) / len(plan["seeds"])}
            for key, values in aggregate.items()], key=lambda row: -row["exclusive_total_share_percent_mean"])[:20]
        summary.append(item)
    dump(out / "profile_summary.json", summary)
    flat_rows = [{"method": r["method"], "seed": r["seed"],
                  "uninstrumented_wall_s": r["uninstrumented_wall_s"], "instrumented_wall_s": r["instrumented_wall_s"],
                  "profiler_overhead_percent": r["profiler_overhead_percent"],
                  **r["bucket_exclusive_percent"]} for r in rows]
    with (out / "profile_seed_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader(); writer.writerows(flat_rows)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
    manifest = {"schema": "particle-filter.profiling-supplement.v1", "status": "completed",
                "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(),
                "git_parent_commit": git.stdout.strip(), "plan_sha256": sha(PLAN), "source_code_sha256": code_hashes,
                "scientific_output_identity_checks": len(rows), "identity_checks_passed": len(rows),
                "argv": ["python", "code/audit_reporting/profile_runner.py", "--out-dir", out.relative_to(REPO).as_posix()],
                "limitations": plan["limitations"],
                "artifact_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}}
    dump(out / "profile_manifest.json", manifest)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
