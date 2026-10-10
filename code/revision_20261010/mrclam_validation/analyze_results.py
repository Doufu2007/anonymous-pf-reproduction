"""Analyse frozen MR.CLAM audit runs without copying third-party raw data.

The analysis unit is a *seed paired within dataset and robot*.  Seeds are
Monte-Carlo repetitions of the same real trajectory, not independent
trajectories.  Bootstrap intervals therefore describe conditional Monte-Carlo
uncertainty; they are not dataset-generalisation intervals or p-values.

Expected layout (the runner's layout is also accepted when ``--runs`` points
at the batch root)::

    dataset-08/robot-1/B-stable/seed-00/raw_metrics.json
    dataset-08/robot-1/B-stable/seed-00/manifest.json
    ...

No MR.CLAM observation or truth arrays are written by this script.  Only
summary tables, validation metadata, and (when matplotlib is available) a
paired-delta PDF are emitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

METHODS = ("B-stable", "RM-stable", "Q-stable", "Q-stable-retain")
PRIMARY = {
    "pre_move_radial_rmse_m": "rmse_position_radial_m",
    "post_assimilation_radial_rmse_m": "post_diagnostic_rmse_position_radial_m",
}
COUNT_FIELDS = (
    "likelihood_evals", "scalar_observation_evals", "candidate_evals",
    "proposal_candidate_attempts", "proposal_chromosome_events",
    "rotation_calls", "crossover_operations", "measurement_bit_observations",
    "probability_entry_updates", "fitness_evals", "accepts", "rejects",
    "resampling_calls", "motion_state_segment_evals",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def discover(runs: Path | Iterable[Path]) -> list[dict]:
    """Discover metrics from one or more run roots and reject duplicate keys."""
    if isinstance(runs, (str, os.PathLike, Path)):
        roots = [Path(runs)]
    else:
        roots = [Path(x) for x in runs]
    if not roots:
        raise ValueError("at least one --runs directory is required")
    resolved_roots = [p.resolve() for p in roots]
    if len(set(resolved_roots)) != len(resolved_roots):
        raise ValueError("--runs contains the same directory more than once")
    rows = []
    seen = set()
    for root in resolved_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"run directory not found: {root}")
        for metrics_path in sorted(root.rglob("raw_metrics.json")):
            run = metrics_path.parent
            try:
                m = _json(metrics_path)
            except Exception as exc:
                raise ValueError(f"invalid raw metrics: {metrics_path}: {exc}") from exc
            required = {"dataset_id", "robot_id", "seed", "method", *PRIMARY.values()}
            missing = sorted(required - m.keys())
            if missing:
                raise ValueError(f"{metrics_path} missing {missing}")
            if m["method"] not in METHODS:
                raise ValueError(f"unknown method {m['method']} in {metrics_path}")
            key = (int(m["dataset_id"]), int(m["robot_id"]), int(m["seed"]), str(m["method"]))
            if key in seen:
                raise ValueError(f"duplicate run key {key}; --runs roots overlap")
            seen.add(key)
            row = dict(m)
            row["run_dir"] = str(run)
            row["metrics_path"] = str(metrics_path)
            row["manifest_path"] = str(run / "manifest.json")
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no raw_metrics.json below {runs}")
    return rows


def _prediction_metrics(run: Path) -> dict:
    """Recompute endpoint metrics from the portable predictions.csv file."""
    path = run / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    names = set(data.dtype.names or ())
    expected = {"truth_x", "truth_y", "pre_move_x", "pre_move_y",
                "post_move_x", "post_move_y"}
    if not expected <= names:
        raise ValueError(f"{path} missing prediction columns {sorted(expected - names)}")
    def radial(prefix):
        e = np.column_stack([data[f"{prefix}_x"] - data["truth_x"],
                             data[f"{prefix}_y"] - data["truth_y"]])
        return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))
    return {"pre_move_radial_rmse_m": radial("pre_move"),
            "post_assimilation_radial_rmse_m": radial("post_move")}


def _bootstrap_ci(values: np.ndarray, seed: int = 20261010,
                  draws: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return (float(np.quantile(means, .025)), float(np.quantile(means, .975)))


def _effect(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    sd = float(np.std(values, ddof=1))
    return float(np.mean(values) / sd) if sd > 0 else (float("inf") if np.mean(values) else 0.)


def _write_csv(path: Path, rows: Iterable[dict]):
    rows = list(rows)
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def validate_runs(rows: list[dict], expected: int = 240) -> dict:
    """Validate manifests, file hashes, predictions and all endpoint metrics."""
    manifest_total = manifest_ok = hash_ok = metrics_ok = 0
    failures = []
    keys = [(int(r["dataset_id"]), int(r["robot_id"]), int(r["seed"]), str(r["method"])) for r in rows]
    duplicate_keys = sorted({k for k in keys if keys.count(k) > 1})
    expected_layout = False
    if expected == 240:
        expected_keys = {(dataset, 1, seed, method)
                         for dataset in (8, 9) for seed in range(30) for method in METHODS}
        expected_layout = set(keys) == expected_keys and len(keys) == len(expected_keys)
        if not expected_layout:
            failures.append({"kind": "unexpected_240_run_layout",
                             "missing": [list(k) for k in sorted(expected_keys - set(keys))[:20]],
                             "extra": [list(k) for k in sorted(set(keys) - expected_keys)[:20]]})
    for row in rows:
        run = Path(row["run_dir"])
        manifest_path = run / "manifest.json"
        manifest_total += 1
        if not manifest_path.exists():
            failures.append({"run": str(run), "kind": "missing_manifest"}); continue
        try:
            manifest = _json(manifest_path)
            if manifest.get("status") != "completed":
                raise ValueError(f"status={manifest.get('status')}")
            for key in ("dataset_id", "robot_id", "seed", "method"):
                if manifest.get(key) != row[key]:
                    raise ValueError(f"manifest {key} disagrees")
            manifest_ok += 1
            file_hashes = manifest.get("files_sha256", {})
            missing_or_bad = []
            for name, digest in file_hashes.items():
                p = run / name
                if not p.exists() or sha256(p) != digest:
                    missing_or_bad.append(name)
            if missing_or_bad:
                raise ValueError(f"manifest file hashes invalid: {missing_or_bad}")
            hash_ok += 1
            recomputed = _prediction_metrics(run)
            raw_keys = {"pre_move_radial_rmse_m": "rmse_position_radial_m",
                        "post_assimilation_radial_rmse_m": "post_diagnostic_rmse_position_radial_m"}
            for key, value in recomputed.items():
                raw_key = raw_keys[key]
                if not math.isclose(float(row[raw_key]), value, rel_tol=0., abs_tol=1e-10):
                    raise ValueError(f"{raw_key} differs raw={row[raw_key]} recomputed={value}")
            metrics_ok += 1
        except Exception as exc:
            failures.append({"run": str(run), "kind": "validation", "message": str(exc)})
    return {"runs_discovered": len(rows), "expected_runs": expected,
            "manifest_count": manifest_total, "manifest_valid": manifest_ok,
            "file_hashes_valid": hash_ok, "metrics_recomputed": metrics_ok,
            "duplicate_run_keys": [list(k) for k in duplicate_keys],
            "expected_240_layout": expected_layout if expected == 240 else None,
            "all_expected_runs_present": len(rows) == expected and not duplicate_keys and (expected != 240 or expected_layout),
            "all_manifests_valid": manifest_total == expected and manifest_ok == expected,
            "all_hashes_valid": hash_ok == expected,
            "all_metrics_recomputed": metrics_ok == expected,
            "failures": failures}


def compare_replay_dirs(first: Path, second: Path) -> dict:
    """Compare two fixed-seed replay trees, excluding timing/provenance files.

    This is deliberately opt-in because replay directories are produced by the
    runner, not by the analysis step. Numeric arrays and non-timing metrics are
    compared byte-for-byte after loading; runtime fields and manifests are
    excluded because wall-clock and file hashes legitimately differ.
    """
    # Per-event elapsed timings are hardware/scheduler dependent and are not
    # scientific arrays. Runtime remains a secondary profiling outcome, while
    # replay equality concerns estimates, particles, weights and counters.
    first = Path(first).resolve()
    second = Path(second).resolve()
    if first == second:
        raise ValueError(f"refusing replay self-comparison: {first}")
    if not first.is_dir() or not second.is_dir():
        raise FileNotFoundError(f"replay directory missing: {first} or {second}")
    scientific_suffixes = {".npy", ".npz"}
    timing_arrays = {"step_elapsed_s.npy"}
    files_a = {p.relative_to(first).as_posix(): p for p in first.rglob("*")
               if p.is_file() and p.suffix.lower() in scientific_suffixes
               and p.name not in timing_arrays}
    files_b = {p.relative_to(second).as_posix(): p for p in second.rglob("*")
               if p.is_file() and p.suffix.lower() in scientific_suffixes
               and p.name not in timing_arrays}
    compared, exact, failures = 0, 0, []
    for rel in sorted(set(files_a) | set(files_b)):
        if rel not in files_a or rel not in files_b:
            failures.append({"file": rel, "kind": "missing_in_replay"}); continue
        try:
            a, b = np.load(files_a[rel], allow_pickle=False), np.load(files_b[rel], allow_pickle=False)
            if isinstance(a, np.lib.npyio.NpzFile):
                ka, kb = set(a.files), set(b.files)
                same = ka == kb and all(np.array_equal(a[k], b[k], equal_nan=True) for k in ka)
                a.close(); b.close()
            else:
                same = np.array_equal(a, b, equal_nan=True)
            compared += 1
            if same: exact += 1
            else: failures.append({"file": rel, "kind": "array_mismatch"})
        except Exception as exc:
            failures.append({"file": rel, "kind": "load_error", "message": str(exc)})
    # Metrics are compared after removing known timing fields. This catches a
    # changed scientific result while allowing harmless wall-clock variation.
    metric_a = {p.relative_to(first).as_posix(): p for p in first.rglob("raw_metrics.json")}
    metric_b = {p.relative_to(second).as_posix(): p for p in second.rglob("raw_metrics.json")}
    metric_compared = metric_exact = 0
    timing_keys = {"runtime_s", "initialization_runtime_s"}
    for rel in sorted(set(metric_a) | set(metric_b)):
        if rel not in metric_a or rel not in metric_b:
            failures.append({"file": rel, "kind": "missing_metrics_in_replay"}); continue
        x, y = _json(metric_a[rel]), _json(metric_b[rel])
        x, y = {k: v for k, v in x.items() if k not in timing_keys}, {k: v for k, v in y.items() if k not in timing_keys}
        metric_compared += 1
        if x == y: metric_exact += 1
        else: failures.append({"file": rel, "kind": "metrics_mismatch"})
    return {"first": str(first), "second": str(second), "scientific_arrays_compared": compared,
            "scientific_arrays_exact": exact, "non_timing_metrics_compared": metric_compared,
            "non_timing_metrics_exact": metric_exact, "exact": not failures,
            "failures": failures}


def analyse(rows: list[dict], bootstrap_draws: int = 20000):
    grouped = {}
    for row in rows:
        key = (int(row["dataset_id"]), int(row["robot_id"]))
        grouped.setdefault(key, {})[(row["method"], int(row["seed"]))] = row
    primary_rows, delta_rows, secondary_rows, count_rows = [], [], [], []
    for (dataset, robot), records in sorted(grouped.items()):
        for method in METHODS:
            vals = [r for (m, _), r in records.items() if m == method]
            for phase, field in PRIMARY.items():
                x = np.array([float(r[field]) for r in vals])
                primary_rows.append({"dataset_id": dataset, "robot_id": robot,
                    "method": method, "phase": phase, "n_seeds": len(x),
                    "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x)>1 else 0.,
                    "ci95_low": _bootstrap_ci(x, 20261010 + dataset)[0],
                    "ci95_high": _bootstrap_ci(x, 20261010 + dataset)[1]})
            if vals:
                for field, label in (("mean_ess_fraction", "ess_fraction"),
                                     ("mean_distinct", "distinct_particles"),
                                     ("runtime_s", "runtime_s")):
                    x = np.array([float(r[field]) for r in vals])
                    secondary_rows.append({"dataset_id": dataset, "robot_id": robot,
                        "method": method, "metric": label, "n_seeds": len(x),
                        "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x)>1 else 0.,
                        "min": float(np.min(x)), "max": float(np.max(x))})
                c = {field: int(sum(int(r.get(field, 0)) for r in vals)) for field in COUNT_FIELDS}
                c_mean = {f"mean_per_seed_{field}": (float(c[field] / len(vals)) if vals else float("nan"))
                          for field in COUNT_FIELDS}
                count_rows.append({"dataset_id": dataset, "robot_id": robot,
                    "method": method, "n_seeds": len(vals), **c, **c_mean,
                    "counter_definition": "sum of per-run algorithm counters; seed is MC repetition"})
        baseline = {int(seed): row for (method, seed), row in records.items() if method == "B-stable"}
        for method in METHODS[1:]:
            for phase, field in PRIMARY.items():
                paired = [(seed, float(records[(method, seed)][field]) - float(baseline[seed][field]))
                          for seed in baseline if (method, seed) in records]
                d = np.array([x[1] for x in paired], dtype=float)
                lo, hi = _bootstrap_ci(d, 20301010 + dataset + len(phase), bootstrap_draws)
                delta_rows.append({"dataset_id": dataset, "robot_id": robot, "method": method,
                    "phase": phase, "baseline": "B-stable", "n_pairs": len(d),
                    "mean_delta_m": float(np.mean(d)) if len(d) else float("nan"),
                    "sd_delta_m": float(np.std(d, ddof=1)) if len(d)>1 else float("nan"),
                    "ci95_low_m": lo, "ci95_high_m": hi,
                    "dz_mc_descriptive": _effect(d),
                    "interpretation": "conditional Monte-Carlo paired delta; not a p-value or dataset-generalisation CI",
                    "seed_values": json.dumps({str(k): float(v) for k, v in paired}, sort_keys=True)})
    return primary_rows, delta_rows, secondary_rows, count_rows


def make_figure(delta_rows: list[dict], path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {"created": False, "reason": "matplotlib unavailable"}
    methods = METHODS[1:]
    datasets = sorted({int(r["dataset_id"]) for r in delta_rows})
    phases = list(PRIMARY)
    fig, axes = plt.subplots(1, len(phases), figsize=(10, 4.2), squeeze=False)
    colors = {"RM-stable": "#0072B2", "Q-stable": "#D55E00", "Q-stable-retain": "#009E73"}
    for ax, phase in zip(axes[0], phases):
        for j, dataset in enumerate(datasets):
            for i, method in enumerate(methods):
                r = next((x for x in delta_rows if int(x["dataset_id"]) == dataset and
                          x["method"] == method and x["phase"] == phase), None)
                if not r: continue
                y = float(r["mean_delta_m"]); lo = float(r["ci95_low_m"]); hi = float(r["ci95_high_m"])
                x = j * 4 + i
                ax.errorbar(x, y, yerr=[[y-lo], [hi-y]], fmt="o", color=colors[method], capsize=3)
        ax.axhline(0., color="black", linewidth=.8)
        ax.set_xticks([j * 4 + 1 for j in range(len(datasets))])
        ax.set_xticklabels([f"Dataset {d}" for d in datasets])
        ax.set_title(phase.replace("_", " "))
        ax.set_ylabel("Mean paired RMSE delta vs B-stable (m)")
        ax.grid(axis="y", alpha=.25)
    handles = [plt.Line2D([0], [0], marker="o", color=colors[m], linestyle="", label=m) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.text(.5, .01, "Points and 95% bootstrap intervals are conditional on paired seeds within each dataset; seeds are not independent trajectories.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .9))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"created": True, "path": str(path), "sha256": sha256(path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="one or more disjoint run roots; duplicate run keys are rejected")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expected-runs", type=int, default=240)
    ap.add_argument("--bootstrap-draws", type=int, default=20000)
    ap.add_argument("--figure", type=Path, default=None)
    ap.add_argument("--replay-a", type=Path, default=None,
                    help="optional seed-0 replay directory for exact array comparison")
    ap.add_argument("--replay-b", type=Path, default=None,
                    help="second seed-0 replay directory")
    args = ap.parse_args(argv)
    rows = discover(args.runs)
    validation = validate_runs(rows, args.expected_runs)
    primary, deltas, secondary, counts = analyse(rows, args.bootstrap_draws)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "primary_summary.csv", primary)
    _write_csv(args.out_dir / "paired_deltas.csv", deltas)
    _write_csv(args.out_dir / "secondary_summary.csv", secondary)
    _write_csv(args.out_dir / "audit_counts.csv", counts)
    figure_path = args.figure or (Path(__file__).resolve().parents[4] / "category4-journal-draft" / "figs" / "mrclam-realdata-audit.pdf")
    figure = make_figure(deltas, figure_path)
    if (args.replay_a is None) != (args.replay_b is None):
        raise SystemExit("--replay-a and --replay-b must be supplied together")
    try:
        replay = compare_replay_dirs(args.replay_a, args.replay_b) if args.replay_a else None
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    report = {"schema": "mrclam.analysis.v1", "analysis_unit": "seed paired within dataset/robot",
              "seeds_are_independent_trajectories": False, "bootstrap": "conditional Monte-Carlo over paired seeds",
              "p_values_or_significance_claims": False, "validation": validation, "replay_validation": replay,
              "figure": figure,
              "rows": {"primary": len(primary), "paired_delta": len(deltas), "secondary": len(secondary), "audit_counts": len(counts)},
              "source_runs": [str(Path(p).resolve()) for p in args.runs]}
    (args.out_dir / "analysis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if replay is not None and not replay["exact"]:
        raise SystemExit("replay validation failed; inspect analysis_report.json")
    if not all(validation[k] for k in ("all_expected_runs_present", "all_manifests_valid", "all_hashes_valid", "all_metrics_recomputed")):
        raise SystemExit("validation failed; inspect analysis_report.json")


if __name__ == "__main__":
    main()
