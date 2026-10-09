# EC q-zero interpreted variant

This directory contains the single declared E1 revision. It preserves the
literal historical implementation in `code/source_aligned_audit/` and changes
only the best-mask mapping used by the new swarm update:

```text
q = P(bit0), bit1 iff q < u
literal:     q_best = alpha*D + (1-alpha)*(1-D)
consistent:  q_best = alpha*(1-D) + (1-alpha)*D
```

The consistent implementation is an explicitly labelled interpretation
variant. It is not a claim about the original authors' intent and is not a
complete reconstruction of the source tracker or quantum implementation.

`test_interpreted.py` is the meaningful unit/property/contract suite. The
red logs retain the missing-module and pre-fix direction failures; the green
log records the ten passing tests. `run_revision.py` verifies the frozen
protocol, historical input hashes, shapes and comparator arrays before running
the one new method. It reuses the archived literal, exact and systematic raw
arrays and never ranks their historical timings against the new timing.

`analyze_revision.py` consumes a completed batch and reports the frozen
dataset-level (30 datasets × 20 updates) RMSE comparisons, 20,000 whole-dataset
bootstrap resamples, 50,000 sign flips and Holm family-3 adjustment. It also
exports secondary metrics, operation counters and input/output hashes.

The parent task released the M1-before-M2 gate before execution. The complete
scientific batch is `runs-consistent-30-v2/`; `runs-consistent-30/` preserves a
launcher/environment-capture failure before any update was executed. The
NumPy `show_config` capture fix is documented in `completion-report.md` and
does not change the selection algorithm or frozen matrix.

Two independent process replays are `replay-consistent-a/` and
`replay-consistent-b/`. `replay-comparison.json` confirms identical scientific
outputs in the recorded environment. `analysis-consistent-30-v2/` retains the
analysis; `readonly-verification.json` reports the full checksum/array/fitness
audit. Local manifests are the original execution evidence; publication path
sanitization is handled separately without overwriting these facts.
