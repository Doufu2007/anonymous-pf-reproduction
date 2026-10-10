# Source-strict implementation status

## Completed in this revision

- Added `code/qm_hpf_reproduction/source_strict/` with a one-dimensional QM-HPF source inventory, configuration gate, trace schema and fail-closed runner.
- Added `code/qgpf_reproduction/source_strict/` with the source constants $N=400$, $m=10$, $t=4$, source inventory, configuration gate and trace schema.
- Added `code/source_aligned_audit/source_strict/` with the full 180-step EC-QPF source inventory, source-literal Eq. (22) contract, dimensional-audit fields, configuration gate and trace schema.
- Added `metadata/source-material-request-templates-2026-10-10.md` for requesting missing source code, raw data and parameter definitions.

## Current execution status

All three strict runners are intentionally blocked by unresolved source evidence. The existing outputs remain labelled `QM-HPF-like`, `QGPF-like` and `EC-QPF module audit`. The strict runners do not import or silently inherit those completion choices.

This is an implementation of the reproduction gate and trace contract. It is not evidence that the original algorithms have already been fully reproduced. A method can be upgraded only after its source contract is closed and its source tables/figures are independently recovered within a preregistered tolerance.
