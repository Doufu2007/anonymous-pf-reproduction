# Particle-filter implementation audit package

This repository contains audit implementations, synthetic benchmark outputs,
configuration contracts and evidence metadata for a focused review of
quantum-inspired particle filtering. It does not provide strict reproductions
of the complete QM-HPF, QGPF or EC-QPF source algorithms.

## Version boundary

Historical baseline: `v1.0.0-submission`; its tag and archived results are retained.
The current revision is `v1.2.0-q2-review`; resolve the exact commit with
`git rev-parse 'v1.2.0-q2-review^{}'`. The preceding audit release
`v1.1.0-audit-update` is retained for comparison.

The manuscript revision dated 2026-10-09 is staged under
`code/revision_20261009/` and `metadata/revision-2026-10-09/`. It contains the
stable/retention QGPF-like audits, the EC-QPF probability-convention audit,
their manifests, raw arrays, analysis tables, operation definitions and
same-environment replay checks. These files are part of the current versioned
release; the manuscript should cite the `v1.2.0-q2-review` commit rather than
the historical `v1.1.0-audit-update` tag.

The update adds the 30-seed EC-QPF selection-module audit, fair-likelihood-cap
code/configuration and analysis evidence, refreshed screening/citation metadata,
operation-count reporting and profiling evidence. Its final commit is identified
by the new Git tag; it is not the historical baseline commit. A Git tag is a
versioned repository reference, not a DOI-backed preservation deposit. No DOI
is asserted by this package.

The retained passage-level semantic audit refers to the manuscript hashes it
originally reviewed. It is explicitly historical after subsequent count and
profiling revisions; its verdicts must not be used to claim that the updated
manuscript passed a new full semantic review.

## Contents

- `code/qm_hpf_reproduction/`: QM-HPF-like audit, source/completion choices,
  analysis code, and archived native outputs.
- `code/qm_hpf_reproduction/native_v4/qm_hpf.py`: native-run implementation
  recorded in the historical v4 manifest. Its SHA-256 is
  `312f3f5d4d9cac0d2758e5a7549ed7aa3fa6bffc832b49fb82ff622c82d89b36`.
- `code/qm_hpf_reproduction/runs/fair-budget-30/`: exploratory 30-seed common
  likelihood-upper-cap comparison, raw outputs, guardrails and analysis. It is
  not an exact equal-realized-cost or equal-wall-clock comparison.
- `code/qgpf_reproduction/`: QGPF-like audit and archived outputs, with explicit
  departures and unresolved source parameters.
- `code/classical_benchmark/`: classical PF reference benchmark.
- `code/source_aligned_audit/`: source-equation EC-QPF one-step selection-module
  audit, contracts, 30 seed datasets and fixed-seed replay. It is separate from
  the sequential PF likelihood-budget experiments.
- `code/audit_reporting/`: operation-count aggregation and descriptive
  three-seed profiling, with retained per-run evidence and measurement limits.
  See `COUNTER_README.md` and `profile_README.md` in that directory.
- `code/revision_20261009/`: revision batches for the QGPF-like numerical and
  retention contracts and the EC-QPF interpreted probability convention.
- `metadata/revision-2026-10-09/`: wrapper provenance for the revision release.
- `metadata/search_ledger.csv`: search and screening ledger.
- `metadata/citation_registry_2026-10-08.json`: historical registry export.
- `metadata/citation_registry_2026-10-09.json`: refreshed registry export.
- `metadata/implementation_audit_2026-10-09.md`: corrected interpretation of
  historical implementations; it does not overwrite their results.
- `metadata/claim_semantic_review_final_2026-10-09.json`: historical audit
  projection retaining IDs, verdicts, original hashes and evidence locators.
  Manuscript prose and third-party source excerpts are excluded.
- `metadata/citation-evidence-2026-10-08/` and
  `metadata/citation-evidence-2026-10-09/`: public bibliographic API metadata
  and search-response snapshots, not article full texts.
- `metadata/evidence-export-provenance.json`: original/export hashes and
  transformations used for public evidence copies.
- `metadata/publication-ethics-review.md`: publication-scope review and limits.

Source-artifact hashes preserve evidence identity; a referenced source basename
does not mean that the source article is redistributed here. No third-party
article PDF/HTML, full-text extract, source-excerpt collection or manuscript
file is included. Public citation metadata contains the bibliographic authors
of cited works; those records are distinct from manuscript author identifiers.

## Environment and reruns

Historical recorded environment: Python 3.13.7, NumPy 2.4.2; the likelihood-cap
supplement also used SciPy 1.17.1. Consult each run's environment/configuration
manifest for its exact scope. Run the following relative commands from the
repository root, using fresh output directories:

```powershell
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:VECLIB_MAXIMUM_THREADS='1'

cd code/qm_hpf_reproduction
python run_reproduction.py --seeds 0:29 --out-dir runs-main-30-rerun
python analyze_results.py runs-main-30-rerun
python -m unittest test_budget_benchmark -v
python run_budget_benchmark.py --plan fair_budget_plan.json --out-dir runs/fair-budget-30-rerun
python analyze_budget_benchmark.py runs/fair-budget-30-rerun

cd ../qgpf_reproduction
python run_qgpf.py --seeds 0:29 --out-dir runs-main-30-rerun

cd ../source_aligned_audit
python -m unittest test_operator -v
python run_audit.py --out runs-main-30-rerun
```

For counts and profiling, run from the repository root. The counter script
reads the historical archives; its optional seed-0 replay checks recorded
outputs. Profiling is a separate descriptive timing supplement and does not
replace the 30-seed results. Set the environment variables shown above before
starting its Python process:

```powershell
python code/audit_reporting/build_audit_counters.py --verify-seed0
python code/audit_reporting/profile_runner.py --out-dir code/audit_reporting/profile-runs-new
python code/audit_reporting/profile_archive_check.py --profile-dir code/audit_reporting/profile-runs-new
```

Detailed definitions and replay boundaries are in
`code/audit_reporting/COUNTER_README.md` and
`code/audit_reporting/profile_README.md`. Profiling reports exclusive helper
time shares with unobserved costs marked unavailable, not complete algorithm
phase times or an equal-wall-clock experiment.

The native QM-HPF-like code snapshot and the budget-aware parent implementation
are different versions. The budget runner refuses to overwrite its archived
bundle. Reruns retain truth for evaluation and must not use it as algorithm input.
All trajectories and observations are generated synthetic data. There are no
participant, clinical, animal, proprietary or personal datasets in the release.

Fixed-seed checks are same-environment implementation checks. They do not show
independent reimplementation, cross-hardware reproducibility or effectiveness
of the original complete source algorithms. Counts and measured wall time use
different units; equal likelihood budgets do not imply equal elapsed costs.

## Privacy and licenses

Release-file contents omit manuscript author identifiers, correspondence,
ORCID, account links and local absolute paths. Repository hosting accounts and
Git history are not anonymized by removing identifiers from file contents;
use an appropriate anonymous review mirror if the venue requires it.

- Original code and configuration: MIT (`LICENSE-CODE`).
- Generated benchmark outputs: CC BY 4.0 (`LICENSE-DATA`).
- Public bibliographic metadata retains the terms of its source providers.
