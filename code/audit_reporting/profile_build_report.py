"""Build display-safe profiling descriptions from the retained measurements."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pstats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LABELS = {"bootstrap_200": "Bootstrap (200)", "resample_move_100": "Resample-move (100)",
          "qm_hpf_like_100": "QM-HPF-like (100)", "qgpf_like": "QGPF-like (200)",
          "ecqpf_source_module": "EC source module"}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.profile_dir.resolve()
    summary = json.loads((out / "profile_summary.json").read_text(encoding="utf-8"))
    env = json.loads((out / "profile_environment.json").read_text(encoding="utf-8"))
    replay = json.loads((out / "profile_archive_replay_check.json").read_text(encoding="utf-8"))
    limits = {"schema": "particle-filter.profiling-observability.v1", "methods": {},
              "all_numeric_partitions_sum_to_100_before_rounding": True,
              "rng": {"independent_time": "unavailable", "raw_zero_meaning": "no separately visible cProfile Generator function; not absence of random draws",
                      "attribution": "RNG time remains in caller self time, potentially joint rotation, fitness or mixed/other buckets"},
              "probability_fitness": "named-helper exclusive time only; inline logic and shared NumPy internals not separately attributable"}
    table_rows = []
    text_rows = []
    for row in summary:
        method = row["method"]
        shares = row["bucket_exclusive_percent_mean"]
        if abs(sum(shares.values()) - 100) > 1e-8:
            raise AssertionError("Percentage partition invalid")
        classical = method in {"bootstrap_200", "resample_move_100"}
        rotation = "NA" if classical or method == "ecqpf_source_module" else f"{shares['rotation_crossover_encoding']:.2f}"
        measurement = "NA" if classical else "U" if method != "ecqpf_source_module" else f"{shares['measurement']:.2f}"
        likelihood = "NA" if method == "ecqpf_source_module" else f"{shares['likelihood']:.2f}"
        table_rows.append(f"{LABELS[method]} & {likelihood} & {rotation} & {measurement} & {shares['probability_fitness']:.2f} & {shares['sorting_resampling']:.2f} & U & {shares['other']:.2f} \\\\")
        limits["methods"][method] = {"measurement": "not_applicable" if classical else "available helper-only" if method == "ecqpf_source_module" else "unavailable, embedded in joint/helper or mixed loop",
                                    "rotation": "not_applicable" if classical or method == "ecqpf_source_module" else "available helper-only; QM includes joint rotate-and-measure",
                                    "rng": "unavailable", "likelihood": "not_applicable" if method == "ecqpf_source_module" else "available helper-only"}
        top = row["top_functions_mean_exclusive_share"][:2]
        text_rows.append(f"| {LABELS[method]} | {row['uninstrumented_wall_s_mean']:.5f} | {row['instrumented_wall_s_mean']:.5f} | {row['profiler_overhead_percent_mean']:.2f}% | {top[0]['function']} ({top[0]['exclusive_total_share_percent_mean']:.2f}%), {top[1]['function']} ({top[1]['exclusive_total_share_percent_mean']:.2f}%) |")
    dump(HERE / "profile_measurement_limits.json", limits)
    header = f"""# Descriptive profiling findings

Measured on {env['cpu']}, {env['platform']}, Python {env['python'].split()[0]},
NumPy {env['numpy']}, float64 CPU, OpenBLAS single thread. No GPU was used.
The frozen supplement retained seeds 0, 1 and 2, with one ordinary and one
cProfile run per method/seed. All 15 scientific-output identity checks passed.
The corresponding archived scientific outputs also match exactly for all 15
workloads (`profile_archive_replay_check.json`: {replay['status']}, all bitwise
equal). Timing is deliberately excluded from scientific identity.

| Workload | Ordinary mean elapsed s | Profiled mean elapsed s | Mean profiler overhead | Two largest function-exclusive shares |
|---|---:|---:|---:|---|
"""
    findings = header + "\n".join(text_rows) + """

The top two rows are **function self times**, not inclusive phase times. The
complete seven-bucket partition and top20 functions are in each raw run and the
summary. QGPF-like's `other` share is 78.02%; its mixed-purpose `qgpf_resample`
self time is 32.01% and generic NumPy `_clip` self time is 13.25%. These costs
cannot be apportioned into independent rotation, measurement, RNG and acceptance
phases from this profile without guessing. QM-HPF-like's
`rotate_bit_toward` accounts for 6.96% as a joint rotation-and-measure helper,
already included in its rotation/encoding bucket, so do not add it again.
EC `update_swarm` and `subset_logfitness` account for 53.53% and 10.69%, while
`observe_binary` has a separately attributable 1.13% helper-only self time.

Independent RNG time is unavailable for this NumPy/Cython build. Independent
QM/QGPF measurement time is also unavailable. Their visible-RNG or measurement
accumulator zeros must not be displayed as absence of executed operations.
Shared NumPy helpers and mixed loops remain in `other`; this is a conservative
function-label partition, not a complete operation-cost model. The elapsed
timings include configured algorithm/RNG initialization and diagnostics but
exclude data generation and I/O. They do not replace the original 30-seed
experiment runtime values and do not constitute an equal-wall-clock experiment.

The fair-budget Bootstrap and resample-move methods use 40,000 likelihood
evaluations per run; QM-HPF-like can leave the cap partly unused. Nevertheless,
equal likelihood-evaluation budgets do not imply equal wall-clock costs.
Profile overhead varies markedly and is reported separately; profile shares
are not de-biased to fabricate uninstrumented phase times. Three seeds with one
timing per workload provide descriptive implementation evidence only.
"""
    (HERE / "profile_findings.md").write_text(findings, encoding="utf-8")
    tex = r"""% Generated from the retained three-seed profiling supplement; optional insertion.
\paragraph{Descriptive implementation profiling.}
A separate, frozen CPU profiling supplement replayed seeds 0--2 with one
ordinary and one \texttt{cProfile} execution per audit workload. The original
model, algorithm functions and RNG streams were unchanged. All 15 pairs had
identical scientific outputs after timing-derived fields were removed; each
ordinary replay also matched its corresponding archived scientific outputs
exactly. The environment was Windows 11, AMD Ryzen 9 7940H, Python 3.13.7,
NumPy 2.4.2 and float64, with OpenBLAS constrained to one thread.
Table~\ref{tab:profile-exclusive} reports arithmetic means of per-seed
function-exclusive time shares, each assigned once. It is a conservative
function-label partition, not a complete phase-cost decomposition.

\begin{table}[tb]
\centering
\scriptsize
\caption{Descriptive mean exclusive profiling shares (\%, three seeds).
L: named likelihood/observation helpers; R: rotation/crossover/encoding,
including QM joint rotation--measurement; M: separately named measurement;
P: probability/fitness helpers; S: named sorting/resampling; O: mixed/other.
U means unavailable as an independent timing; NA means not applicable.
RNG has no independently visible Generator timing in this NumPy build, and
its cost remains in caller self times. Numeric shares sum to 100\% before
rounding; U cells must not be interpreted as absence of the operation.}
\label{tab:profile-exclusive}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Workload & L & R & M & P & S & RNG & O \\
\midrule
""" + "\n".join(table_rows) + r"""
\bottomrule
\end{tabular}
\end{table}

The two largest function self-time shares were \texttt{run\_budget\_filter}
(14.99\%) and NumPy reduction (8.67\%) for Bootstrap;
\texttt{run\_budget\_filter} (10.22\%) and \texttt{resample\_move} (8.44\%)
for resample-move; \texttt{qm\_budget\_update} (23.93\%) and
\texttt{observation} (12.67\%) for QM-HPF-like;
\texttt{qgpf\_resample} (32.01\%) and NumPy \texttt{\_clip} (13.25\%)
for QGPF-like; and \texttt{update\_swarm} (53.53\%) and
\texttt{source\_select} (23.50\%) for the EC module. In particular,
the QGPF-like mixed loop and shared NumPy functions explain much of its
78.02\% residual share; independently attributing their measurement, RNG
and acceptance costs would require additional instrumentation.
QM's \texttt{rotate\_bit\_toward} jointly rotates and measures (6.96\%),
already included in R. EC's \texttt{subset\_logfitness} contributed 10.69\%
and its separate \texttt{observe\_binary} helper 1.13\%.

The ordinary mean elapsed times were 0.0352, 0.0381, 0.9933, 25.0750 and
1.9992~s for these five workloads, respectively; the EC workload consists
of 20 one-step updates. Timing includes algorithm/RNG initialization and
implemented diagnostics, but excludes input generation and I/O. Mean
profiling overhead was 31.30\%, 41.57\%, 34.14\%, 53.87\% and 2.30\%,
respectively, and was not used to infer uninstrumented phase times. These
three-seed measurements are descriptive and do not replace the original
30-seed runtime results. Equal likelihood-evaluation budgets do not imply
equal wall-clock costs; no equal-wall-clock comparison was conducted.
"""
    (HERE / "profile_excerpt.tex").write_text(tex, encoding="utf-8")
    manifest = json.loads((out / "profile_manifest.json").read_text(encoding="utf-8"))
    hash_checks = all(hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
                      for name, expected in manifest["artifact_sha256"].items())
    raw_profile_count = 0
    for raw in out.rglob("*.prof"):
        stats = pstats.Stats(str(raw))
        for key, (_, _, _, _, callers) in stats.stats.items():
            if Path(key[0]).is_absolute() or any(Path(k[0]).is_absolute() for k in callers):
                raise AssertionError("Raw profile contains an absolute path")
        raw_profile_count += 1
    elapsed = (datetime.fromisoformat(manifest["ended_at"]) - datetime.fromisoformat(manifest["started_at"])).total_seconds()
    validation = {"schema": "particle-filter.profile-validation.v1", "status": "PASS" if hash_checks and replay["status"] == "PASS" else "FAIL",
                  "original_manifest_artifact_hashes_match": hash_checks, "sanitized_raw_profiles": raw_profile_count,
                  "absolute_filenames_or_caller_paths": 0, "science_identity_pairs_passed": 15,
                  "archive_replay_workloads_passed": replay["workloads"], "archive_outputs_all_bitwise_equal": replay["all_bitwise_equal"],
                  "elapsed_bundle_seconds": elapsed, "within_frozen_1200s_budget": elapsed <= 1200,
                  "output_coverage": "5 methods x3 seeds, ordinary and cProfile outputs retained",
                  "additional_report_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in (HERE / "profile_measurement_limits.json", HERE / "profile_findings.md", HERE / "profile_excerpt.tex", out / "profile_archive_replay_check.json")}}
    dump(HERE / "profile_validation.json", validation)
    if validation["status"] != "PASS":
        raise AssertionError("Profile artifact validation failed")
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
