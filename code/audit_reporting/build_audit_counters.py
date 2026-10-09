"""Aggregate immutable audit operation evidence, with explicit count provenance.

This reporting supplement never rewrites historical implementations or runs.
Unknown branch counts remain unavailable in the 30-seed aggregate. Optional
seed-0 replay checks use transparent wrappers that consume no random numbers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CODE = REPO / "code"
FIELDS = (
    "likelihood_evaluations", "candidate_evaluations", "proposal_attempts",
    "rotation_bit_routine_calls", "crossover_proposals", "bit_measurements",
    "persistent_probability_entry_updates", "fitness_evaluations",
    "candidate_distance_evaluations", "generation_iterations",
)
DEFINITIONS = {
    "likelihood_evaluations": "One observation-likelihood/log-likelihood evaluated for one particle state; vectorized arrays count their particle entries, not Python calls.",
    "candidate_evaluations": "One newly proposed state or generated binary subset submitted to candidate scoring. Existing cloud states and copied resampled states are excluded. EC-QPF subset scores also appear as fitness_evaluations; columns are not additive.",
    "proposal_attempts": "The historical proposal counter. QM/native and fair-budget count each candidate-generation attempt; QGPF counts each offspring search episode (four candidate generations); EC-QPF has no such historical proposal counter and is unavailable.",
    "rotation_bit_routine_calls": "One classical bit-level rotation routine call, including zero-angle/no-change branches in QM. Fair-budget uses bit_length times recorded rotation_proposals; it is not a physical quantum-gate count. QGPF conditionally calls _rotate_probability, whose 30-seed count was not recorded.",
    "crossover_proposals": "One invocation of the scalar arithmetic crossover proposal; copied offspring/elites do not count. QGPF implementation contains no crossover.",
    "bit_measurements": "One classical Bernoulli bit observation. QM has one draw per rotation-bit routine; QGPF has two bit draws per dimension/bit/candidate generation; EC-QPF has one draw per swarm/subset-bit/generation. No quantum-device measurement is performed.",
    "persistent_probability_entry_updates": "One stored probability entry written by a probability-update operator, excluding initialization. QGPF updates only mismatched bits; EC-QPF updates all swarm entries after each non-stopping generation. QM computes measurement probabilities locally and has no persistent probability state.",
    "fitness_evaluations": "Historical EC-QPF search/objective counter only: one subset score for the swarm or one existing singleton considered for the exact/systematic comparator. The exact comparator uses an argmin-distance scan, not one exponential evaluation per entry. Final returned-subset diagnostic and optimal-fitness summaries are excluded, as in the original counter. These are not observation likelihoods.",
    "candidate_distance_evaluations": "Historical EC-QPF candidate-cloud distance count (one candidate state to observed center). Diagnostic distance recomputations are excluded, as in the original counter.",
    "generation_iterations": "QM/native and fair-budget: one generation for one low-particle candidate; QGPF: one generation of one offspring search; EC-QPF: one full swarm generation. These different work units must not be compared as equal costs.",
}
INPUTS: dict[str, str] = {}
CHECKS: list[dict] = []


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(path: Path) -> str:
    rel = path.relative_to(REPO).as_posix()
    INPUTS[rel] = digest(path)
    return rel


def load_json(path: Path):
    evidence(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv(path: Path) -> list[dict]:
    evidence(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cell(value, status: str, basis: str, source: str) -> dict:
    return {"value": value, "status": status, "basis": basis, "source": source}


def unavailable(basis: str, source: str) -> dict:
    return cell(None, "UNAVAILABLE", basis, source)


def zero(source: str, reason: str = "Operation is absent from this implementation.") -> dict:
    return cell(0, "CONFIRMED_ZERO", reason, source)


def recorded(value, source: str, field: str) -> dict:
    return cell(int(value), "RECORDED", f"Historical raw counter: {field}.", source)


def derived(value, source: str, formula: str) -> dict:
    return cell(int(value), "DERIVED", formula, source)


def verify_hash(path: Path, expected: str, context: str):
    observed = digest(path)
    if observed.lower() != expected.lower():
        raise ValueError(f"Historical code binding mismatch: {context}")
    evidence(path)
    CHECKS.append({"check": "historical_code_hash", "context": context, "status": "PASS", "sha256": observed})


def empty_counts(source: str) -> dict:
    return {f: zero(source) for f in FIELDS}


def make_row(audit: str, method: str, seed: int, updates: int, counts: dict, source: str, notes=None):
    return {"audit": audit, "method": method, "seed": int(seed), "updates": int(updates),
            "counts": counts, "source": source, "notes": notes or []}


def native_rows() -> list[dict]:
    root = CODE / "qm_hpf_reproduction"
    manifest = load_json(root / "runs-main-30-v4/manifest.json")
    source = root / "native_v4/qm_hpf.py"
    verify_hash(source, manifest["code_sha256"], "native_v4")
    code_ref = source.relative_to(REPO).as_posix()
    raw_path = root / "runs-main-30-v4/raw_metrics.csv"
    rows = load_csv(raw_path)
    raw_ref = raw_path.relative_to(REPO).as_posix()
    horizon, n = manifest["model"]["horizon"], manifest["filter"]["particle_count"]
    bit_length, generations = manifest["filter"]["bit_length"], manifest["filter"]["generation_count"]
    result = []
    for r in rows:
        counts = empty_counts(code_ref)
        for out, field in (("likelihood_evaluations", "likelihood_evals"),
                           ("candidate_evaluations", "candidate_evals"),
                           ("proposal_attempts", "proposal_attempts")):
            counts[out] = recorded(r[field], raw_ref, field)
        cand = int(r["candidate_evals"])
        counts["generation_iterations"] = derived(cand, code_ref, "One generation for one low particle creates/evaluates one candidate; equals candidate_evals.")
        if r["method"] == "qm_hpf":
            for key in ("rotation_bit_routine_calls", "crossover_proposals", "bit_measurements"):
                counts[key] = unavailable("Stochastic rotation/crossover branch counts were not retained for the historical 30-seed run.", raw_ref)
        elif r["method"] == "rotation_only":
            counts["rotation_bit_routine_calls"] = derived(bit_length*cand, code_ref, "bit_length * candidate_evals; includes zero-angle bit routine calls.")
            counts["bit_measurements"] = derived(bit_length*cand, code_ref, "One Bernoulli draw per bit rotation routine call.")
        elif r["method"] == "crossover_only":
            counts["crossover_proposals"] = derived(cand, code_ref, "Each candidate is produced by crossover_candidate.")
        # Unlimited native inner loops: additional original-state evals equal low-particle episodes.
        original = int(r["likelihood_evals"]) - horizon*n - 2*cand
        if original < 0 or (cand and original*generations != cand):
            raise ValueError("Native likelihood/candidate accounting identity failed")
        result.append(make_row("native", r["method"], int(r["seed"]), horizon, counts, raw_ref))
    CHECKS.append({"check": "native_likelihood_identity", "rows": len(rows), "status": "PASS",
                   "formula": "L = N*T + 2*C + C/G for non-bootstrap unlimited runs"})
    return result


def fair_rows() -> list[dict]:
    root = CODE / "qm_hpf_reproduction"
    raw_path = root / "runs/fair-budget-30/raw_metrics.csv"
    rows = load_csv(raw_path)
    result = []
    for r in rows:
        seed, method = int(r["seed"]), r["method"]
        run = root / f"runs/fair-budget-30/runs/seed-{seed:02d}/{method}"
        cfg = load_json(run / "config.json")
        manifest = load_json(run / "manifest.json")
        raw = load_json(run / "raw_metrics.json")
        for name in ("budget_benchmark.py", "qm_hpf.py"):
            verify_hash(root/name, manifest["code_sha256"][name], f"fair-budget seed={seed} method={method} {name}")
        for name in ("raw_metrics.json", "step_likelihood_evals.npy", "step_proposal_attempts.npy"):
            verify_hash(run/name, manifest["files_sha256"][name], f"fair-budget artifact {seed}/{method}/{name}")
        ref = (run / "raw_metrics.json").relative_to(REPO).as_posix()
        counts = empty_counts("code/qm_hpf_reproduction/budget_benchmark.py")
        for out, field in (("likelihood_evaluations", "likelihood_evals"), ("candidate_evaluations", "candidate_evals"),
                           ("proposal_attempts", "proposal_attempts"), ("crossover_proposals", "crossover_proposals")):
            if int(r[field]) != raw[field]:
                raise ValueError("Fair-budget aggregate/raw mismatch")
            counts[out] = recorded(raw[field], ref, field)
        rotations = raw["rotation_proposals"]
        bits = cfg["filter"]["bit_length"]
        counts["rotation_bit_routine_calls"] = derived(bits*rotations, ref, "bit_length * recorded rotation_proposals; includes no-change bit routine calls.") if rotations else zero(ref)
        counts["bit_measurements"] = derived(bits*rotations, ref, "One Bernoulli draw for each bit rotation routine invocation.") if rotations else zero(ref)
        gen = raw["candidate_evals"] if method == "qm_hpf_like_100" else 0
        counts["generation_iterations"] = derived(gen, ref, "One low-particle generation per QM candidate; classical vectorized move is not a QGA generation.") if gen else zero(ref)
        likelihood_steps = np.load(run/"step_likelihood_evals.npy")
        proposal_steps = np.load(run/"step_proposal_attempts.npy")
        if int(likelihood_steps.sum()) != raw["likelihood_evals"] or int(proposal_steps.sum()) != raw["proposal_attempts"]:
            raise ValueError("Fair-budget step sums mismatch")
        if method == "qm_hpf_like_100" and raw["rotation_proposals"] + raw["crossover_proposals"] != raw["candidate_evals"]:
            raise ValueError("Fair-budget proposal branch sum mismatch")
        expected = cfg["particle_count"]*cfg["model"]["horizon"] + raw["candidate_evals"]
        if expected != raw["likelihood_evals"] or np.max(likelihood_steps) > 200:
            raise ValueError("Fair-budget likelihood accounting/cap mismatch")
        result.append(make_row("fair-budget", method, seed, cfg["model"]["horizon"], counts, ref))
    CHECKS.append({"check": "fair_budget_raw_step_sums_cap_and_cached_identity", "rows": len(rows), "status": "PASS"})
    return result


def qgpf_rows() -> list[dict]:
    root = CODE / "qgpf_reproduction"
    manifest = load_json(root / "runs-main-30/manifest.json")
    verify_hash(root/"qgpf.py", manifest["code_sha256"], "QGPF-like")
    code_ref = "code/qgpf_reproduction/qgpf.py"
    path = root / "runs-main-30/raw_metrics.csv"
    rows = load_csv(path)
    ref = path.relative_to(REPO).as_posix()
    horizon = manifest["model"]["horizon"]
    n, bits, gens = (manifest["filter"][f] for f in ("particle_count", "bit_length", "generation_count"))
    dim = len(manifest["filter"]["initial_mean"])
    elite = max(1, min(n-1, int(round(manifest["filter"]["elite_fraction"]*n))))
    result = []
    for r in rows:
        counts = empty_counts(code_ref)
        for out, field in (("likelihood_evaluations", "likelihood_evals"), ("candidate_evaluations", "candidate_evals"), ("proposal_attempts", "proposal_attempts")):
            counts[out] = recorded(r[field], ref, field)
        if r["method"] == "qgpf":
            candidates = int(r["candidate_evals"])
            counts["generation_iterations"] = derived(candidates, code_ref, "One candidate per offspring generation = T*(N-elite_n)*G.")
            counts["bit_measurements"] = derived(2*dim*bits*candidates, code_ref, "Two Bernoulli bit observations per coordinate/bit/candidate generation = 2*d*b*C.")
            for key in ("rotation_bit_routine_calls", "persistent_probability_entry_updates"):
                counts[key] = unavailable("Conditional mismatched-bit rotation counts were not retained in historical 30-seed logs.", ref)
            if candidates != horizon*(n-elite)*gens or int(r["proposal_attempts"]) != horizon*(n-elite):
                raise ValueError("QGPF generation/offspring identity failed")
            if int(r["likelihood_evals"]) != horizon*(2*n+(n-elite)*(gens+1)):
                raise ValueError("QGPF likelihood identity failed")
        result.append(make_row("QGPF-like", r["method"], int(r["seed"]), horizon, counts, ref,
                              ["Historical proposal_attempts count offspring search episodes, not individual candidate generations."]))
    CHECKS.append({"check": "qgpf_likelihood_offspring_generation_identities", "rows": len(rows), "status": "PASS"})
    return result


def ec_rows() -> list[dict]:
    root = CODE / "source_aligned_audit/runs-main-30"
    result = []
    for run in sorted(root.glob("seed-*")):
        cfg = load_json(run/"config.json")["operator"]
        manifest = load_json(run/"manifest.json")
        for c in manifest["code"]["files"]:
            verify_hash(run/c["path"], c["sha256"], f"EC historical snapshot {run.name}/{c['path']}")
        path = run/"raw_metrics.jsonl"
        verify_hash(path, manifest["artifacts"]["raw_metrics"]["sha256"], f"EC raw metrics {run.name}")
        ref = evidence(path)
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        history = {h["update"]: h["history"] for h in load_json(run/"source_generation_history.json")}
        grouped = defaultdict(list)
        for r in raw:
            grouped[r["method"]].append(r)
        for method, rows in grouped.items():
            counts = empty_counts("code/source_aligned_audit/ecqpf_operator.py")
            for key in ("likelihood_evaluations", "fitness_evaluations", "candidate_distance_evaluations"):
                counts[key] = recorded(sum(r[key] for r in rows), ref, key)
            if method == "source_ecqpf_operator":
                counts["candidate_evaluations"] = derived(sum(r["fitness_evaluations"] for r in rows), ref, "One generated binary mask for every recorded swarm fitness evaluation.")
                counts["proposal_attempts"] = unavailable("Swarm subset search has no historical proposal_attempts counter; use separate subset/generation counts.", ref)
                counts["bit_measurements"] = recorded(sum(r["binary_dimensions_processed"] for r in rows), ref, "binary_dimensions_processed (one Bernoulli bit observation per entry)")
                counts["generation_iterations"] = recorded(sum(r["generations"] for r in rows), ref, "generations")
                prob_updates = 0
                for r in rows:
                    hist = history[r["update"]]
                    if len(hist) != r["generations"] or r["fitness_evaluations"] != cfg["swarm_size"]*len(hist):
                        raise ValueError("EC fitness/history identity failed")
                    if r["binary_dimensions_processed"] != cfg["swarm_size"]*r["ellipse_count"]*len(hist):
                        raise ValueError("EC binary dimension identity failed")
                    # Recover the literal early stopping decision, including a break at Gmax.
                    unchanged, global_score = 0, -np.inf
                    for _, score, _, _ in hist:
                        unchanged = 0 if score > global_score else unchanged+1
                        global_score = max(global_score, score)
                    stopped_before_update = unchanged >= max(1, cfg["iteration_max"]//2)
                    update_calls = len(hist)-int(stopped_before_update)
                    prob_updates += cfg["swarm_size"]*r["ellipse_count"]*update_calls
                counts["persistent_probability_entry_updates"] = derived(prob_updates, ref,
                    "Sum swarm_size*M*(G - I(terminal no-improvement stop)); terminal stop recovered from generation history. The max-iteration terminal generation still updates q unless the no-improvement break fires.")
            result.append(make_row("EC-QPF module", method, int(run.name.split("-")[1]), len(rows), counts, ref,
                ["Recorded fitness/distance counts exclude final diagnostic summaries; one-step selection module, not a complete PF."]))
    CHECKS.append({"check": "ec_history_fitness_binary_accounting", "seed_datasets": len(list(root.glob('seed-*'))), "status": "PASS"})
    return result


def classical_rows() -> list[dict]:
    root = CODE / "classical_benchmark"
    code_ref = evidence(root/"benchmark_pf.py")
    rows = load_csv(root/"benchmark_results/classical_pf_benchmark.csv")
    result = []
    for r in rows:
        counts = empty_counts(code_ref)
        if r["method"] == "EKF":
            counts["likelihood_evaluations"] = zero(code_ref, "EKF computes Gaussian linearization/gain, not a particle observation-likelihood evaluation.")
        else:
            counts["likelihood_evaluations"] = derived(100*int(r["n"]), code_ref,
                "T*N, one likelihood for each predicted particle per step, T=100. Historical CSV has no source-code hash binding: derived from supplied current benchmark implementation, not an instrumented historical count.")
        result.append(make_row("classical contextual", f"{r['method']}-N{r['n']}", int(r["seed"]), 100, counts,
                              "code/classical_benchmark/benchmark_results/classical_pf_benchmark.csv",
                              ["Historical benchmark lacks recorded code hash; current-code derivation is explicitly unbound to the historical run."]))
    return result


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["audit"], r["method"])].append(r)
    out = []
    for (audit, method), records in groups.items():
        counts = {}
        for field in FIELDS:
            cells = [r["counts"][field] for r in records]
            vals = [c["value"] for c in cells]
            statuses = sorted({c["status"] for c in cells})
            if any(v is None for v in vals):
                counts[field] = {"mean": None, "sd": None, "min": None, "max": None,
                                 "status": "UNAVAILABLE", "observed_run_count": sum(v is not None for v in vals)}
            else:
                arr = np.array(vals, dtype=float)
                counts[field] = {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(vals)>1 else 0.,
                                 "min": int(min(vals)), "max": int(max(vals)), "status": "+".join(statuses), "observed_run_count": len(vals)}
        out.append({"audit": audit, "method": method, "seed_datasets": len(records),
                    "updates_per_seed": sorted({r["updates"] for r in records}), "counts": counts})
    return out


def import_file(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def replay_seed0() -> dict:
    checks = []
    for family, module, method, rng_seed in (
        ("native", import_file(CODE/"qm_hpf_reproduction/native_v4/qm_hpf.py", "counter_native_v4"), "qm_hpf", 3000009),
        ("QGPF-like", import_file(CODE/"qgpf_reproduction/qgpf.py", "counter_qgpf"), "qgpf", 2000006),
    ):
        log = defaultdict(int)
        def wrap(name, key):
            original = getattr(module, name)
            def counted(*args, **kwargs):
                log[key] += 1
                return original(*args, **kwargs)
            setattr(module, name, counted)
        if family == "native":
            root = CODE/"qm_hpf_reproduction/runs-main-30-v4"
            for name, key in (("quantum_rotation_candidate", "rotation_proposals"), ("crossover_candidate", "crossover_proposals"), ("rotate_bit_toward", "rotation_bit_routine_calls")):
                wrap(name, key)
        else:
            root = CODE/"qgpf_reproduction/runs-main-30"
            wrap("_rotate_probability", "rotation_bit_routine_calls")
        input_paths = [root/"truth_seed0.npy", root/"observations_seed0.npy", root/f"estimate_{method}_seed0.npy"]
        input_hashes = {evidence(path): digest(path) for path in input_paths}
        truth, obs, expected = [np.load(path) for path in input_paths]
        result = module.run_filter(obs, truth, np.random.default_rng(rng_seed), method)
        np.testing.assert_array_equal(result["estimate"], expected)
        row = next(r for r in load_csv(root/"raw_metrics.csv") if int(r["seed"]) == 0 and r["method"] == method)
        for key in ("likelihood_evals", "candidate_evals", "proposal_attempts", "accepts"):
            if result[key] != int(row[key]):
                raise ValueError(f"Seed-0 wrapper changed historical counter: {family}/{key}")
        if family == "native":
            if log["rotation_proposals"] + log["crossover_proposals"] != result["candidate_evals"] or log["rotation_bit_routine_calls"] != 16*log["rotation_proposals"]:
                raise ValueError("Seed-0 native branch identity failed")
            log["bit_measurements"] = log["rotation_bit_routine_calls"]
            log["persistent_probability_entry_updates"] = 0
        else:
            log["persistent_probability_entry_updates"] = log["rotation_bit_routine_calls"]
            log["bit_measurements"] = 2*4*10*result["candidate_evals"]
        checks.append({"audit": family, "method": method, "seed": 0, "status": "PASS",
                       "algorithm_rng_seed": rng_seed,
                       "input_sha256": input_hashes,
                       "source_sha256": digest(Path(module.__file__)),
                       "estimate_exactly_matches_historical_array": True,
                       "historical_counters_match": True,
                       "instrumented_extra_counts": dict(log),
                       "scope": "Supplementary seed-0 transparent-wrapper replay only; never used to fill unavailable historical 30-seed means."})
    return {"schema": "particle_filter.audit_counter_replay.v1", "checks": checks,
            "wrapper_contract": "Wrappers increment integers and call unchanged functions; do not draw random numbers or modify historical files; runtime from this check is not reported as an uninstrumented benchmark."}


def format_cell(item: dict, scale: int = 1000) -> str:
    if item["mean"] is None:
        return r"\textit{unavailable}"
    value = item["mean"] / scale
    marker = r"^{\dagger}" if "DERIVED" in item["status"] else ""
    text = f"{value:,.1f}" if value % 1 else f"{value:,.0f}"
    return f"${text}{marker}$"


def write_tables(aggregates: list[dict], out: Path):
    selected = [r for r in aggregates if r["audit"] != "classical contextual" and not (r["audit"] == "native" and r["method"] in {"mh", "rotation_only", "crossover_only"})]
    labels = {
        ("native", "bootstrap"): "Native Bootstrap PF", ("native", "qm_hpf"): "Native QM-HPF-like",
        ("fair-budget", "bootstrap_200"): "Budget Bootstrap PF", ("fair-budget", "resample_move_100"): "Budget resample-move",
        ("fair-budget", "qm_hpf_like_100"): "Budget QM-HPF-like", ("QGPF-like", "bootstrap"): "QGPF-context Bootstrap",
        ("QGPF-like", "qgpf"): "QGPF-like", ("EC-QPF module", "source_ecqpf_operator"): "EC-QPF selection module",
        ("EC-QPF module", "ellipse_exact_fitness"): "Same-ellipse exact", ("EC-QPF module", "ellipse_systematic_fitness"): "Same-fitness systematic",
        ("EC-QPF module", "full_systematic_likelihood"): "Full-cloud systematic",
    }
    for name, fields, headers in (
        ("audit_counters_table.tex", FIELDS[:6], ["$L$", "$C$", "$P$", "$R_b$", "$X$", "$M_b$"]),
        ("audit_counters_additional_table.tex", FIELDS[6:], ["$A_b$", "$F$", "$D$", "$G$"]),
    ):
        lines = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
                 r"\begin{tabular}{l" + "r"*len(fields) + "}", r"\toprule",
                 "Audit & " + " & ".join(headers) + r" \\", r"\midrule"]
        for r in selected:
            label = labels[(r["audit"], r["method"])]
            lines.append(label + " & " + " & ".join(format_cell(r["counts"][f], 1 if f == "generation_iterations" else 1000) for f in fields) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}",
                  r"\caption{Operation-count means per seed dataset (30 datasets per row); counts are in thousands except $G$, which is unscaled. Each PF trajectory has 200 updates for native/budget audits or 100 updates for QGPF; each EC-QPF seed dataset has 20 independent one-step module updates. $\dagger$ denotes a code-based derivation from retained counters/configuration rather than direct instrumentation. Different columns and audit units do not represent equal costs.}",
                  r"\label{tab:" + ("audit-counts" if name == "audit_counters_table.tex" else "audit-counts-additional") + "}", r"\end{table}"]
        (out/name).write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE/"counter-results")
    parser.add_argument("--verify-seed0", action="store_true")
    args = parser.parse_args()
    rows = native_rows()+fair_rows()+qgpf_rows()+ec_rows()+classical_rows()
    aggregates = aggregate(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    replay = replay_seed0() if args.verify_seed0 else None
    evidence(Path(__file__))
    report = {"schema": "particle_filter.audit_operation_counts.v1", "generated_at": datetime.now(timezone.utc).isoformat(),
              "aggregation": "Mean/SD of total counts within each seed dataset; do not pool 600 module updates as 600 independent datasets.",
              "statuses": {"RECORDED": "Retained historical raw counter, with original counter semantics.",
                           "DERIVED": "Exact code/config/counter formula, explicitly not direct measurement.",
                           "CONFIRMED_ZERO": "No such operator in the supplied implementation.",
                           "UNAVAILABLE": "Counter not retained and cannot be recovered exactly from historical logs."},
              "definitions": DEFINITIONS, "aggregates": aggregates, "seed_rows": rows,
              "input_sha256": dict(sorted(INPUTS.items())), "validation": CHECKS,
              "limitations": ["Counters describe these independent implementations/modules, not the computational cost of the original published algorithms.",
                              "No original algorithm, GPU, quantum circuit, oracle, or physical quantum gate is executed.",
                              "Scalar/vector state likelihoods, binary-mask fitness scores, bit observations, and generations are distinct work units.",
                              "Columns can overlap: generated EC masks are also fitness evaluations; bit rotations and persistent QGPF probability updates can be the same operation.",
                              "Final diagnostic fitness/distance recomputations and initialization are excluded from historical EC counters; counter totals do not measure all work.",
                              "Classical contextual T*N derivations lack an archived source-code hash binding in the original run CSV.",
                              "Missing native/QGPF branch-level means remain unavailable; a seed-0 replay is an additional check, not their replacement."]}
    (args.out/"audit_counters.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (args.out/"audit_counters.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["audit","method","seed_datasets","updates_per_seed","operation","mean","sd","min","max","status","observed_run_count"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in aggregates:
            for operation, value in r["counts"].items():
                writer.writerow({"audit":r["audit"],"method":r["method"],"seed_datasets":r["seed_datasets"],"updates_per_seed":r["updates_per_seed"][0],"operation":operation,**value})
    write_tables(aggregates,args.out)
    if replay:
        (args.out/"seed0_instrumentation_verification.json").write_text(json.dumps(replay,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","seed_rows":len(rows),"aggregate_rows":len(aggregates),"validation_checks":len(CHECKS),"seed0_replay":bool(replay),"out":args.out.name}))


if __name__ == "__main__":
    main()
