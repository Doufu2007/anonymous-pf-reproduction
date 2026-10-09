# Operation-count reporting supplement

Run from the repository root:

```bash
python code/audit_reporting/build_audit_counters.py --verify-seed0
```

The script reads historical code snapshots, configurations, manifests, raw
counters, step arrays, and EC-QPF generation histories. It never modifies the
historical experiments. The default output is `counter-results/` next to the
script. The optional replay verifies the retained native and QGPF seed-0
estimates and counters using transparent wrappers around unchanged functions;
it does not rerun the full 30-seed studies or report instrumented timing.

Outputs:

- `audit_counters.json`: definitions, seed-level counts, means/SD/ranges,
  per-cell provenance, input hashes, and validation results.
- `audit_counters.csv`: exact aggregate values and status for each operation.
- `audit_counters_table.tex` and `audit_counters_additional_table.tex`:
  compact manuscript snippets. Counts are in thousands except the unscaled
  generation column. A dagger means an exact derivation, not instrumentation.
- `seed0_instrumentation_verification.json`: supplementary transparent-wrapper
  replay, when requested. This single-seed evidence never replaces unavailable
  historical branch-level means.

The main snippets include 11 rows across the native, fair-budget, QGPF-like,
and EC-QPF module audits. JSON/CSV also include native ablations and the older
classical contextual benchmark. The contextual benchmark has no archived code
hash binding; its `T*N` likelihood count is explicitly a current-code derivation.

## Symbols and count boundaries

| Symbol | JSON field | Unit |
|---|---|---|
| L | likelihood_evaluations | One state likelihood/log-likelihood |
| C | candidate_evaluations | One generated state/subset submitted to scoring |
| P | proposal_attempts | Original counter: one candidate attempt for QM/budget; one offspring search episode for QGPF |
| R_b | rotation_bit_routine_calls | One classical bit rotation routine invocation, including QM no-change branches |
| X | crossover_proposals | One arithmetic crossover invocation |
| M_b | bit_measurements | One classical Bernoulli bit observation |
| A_b | persistent_probability_entry_updates | One stored probability entry updated, excluding initialization |
| F | fitness_evaluations | Original EC search-score counter, excluding final diagnostics |
| D | candidate_distance_evaluations | Original EC candidate-distance counter, excluding diagnostic recomputation |
| G | generation_iterations | A low-particle generation for QM; an offspring generation for QGPF; a full swarm generation for EC |

`RECORDED` means retained historical raw evidence. `DERIVED` requires an exact
formula in the corresponding seed-row entry. `CONFIRMED_ZERO` identifies an
operator absent from the implementation. `UNAVAILABLE` means it cannot be
recovered exactly from historical records. Missing values are never estimated.

Historical native QM-HPF-like does not retain its stochastic rotation/crossover
split. QGPF does not retain its conditional mismatched-bit rotation count. These
means remain unavailable. QGPF's bit observation count is exactly `2*d*b*C`
because each coordinate-bit is observed twice for every candidate generation.
Fair-budget bit routine/observation counts equal `b*rotation_proposals`.

For EC-QPF, `M_b` is the recorded binary dimensions processed. Probability
updates are derived from the retained early stopping history, including whether
the terminal generation exits before `update_swarm`. At the maximum-generation
limit the final update is performed unless the no-improvement stopping branch
fires. The exact classical comparator's recorded `F=M` represents an argmin
scan over existing singleton entries, without exponentiating every score.

Counts are not additive or universal operation equivalents: an EC generated
mask is also a fitness evaluation; a QGPF rotation also updates a probability
entry. All bit operations are classical simulation. There are no physical
quantum gates, device measurements, or quantum-oracle calls. These measurements
describe bounded independent implementations/modules and cannot determine the
cost or efficacy of the original published algorithms.

The aggregation unit is one seed dataset: a full PF trajectory, or 20
independent one-step EC module updates. EC's 600 updates are not reported as 600
independent paired seed datasets. Scalar state likelihoods, subset fitness
scores, and bit operations have different costs; equal likelihood totals do not
imply equal wall-clock time.
