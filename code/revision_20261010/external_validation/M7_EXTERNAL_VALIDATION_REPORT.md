# M7 execution report: distribution-shift synthetic stress test

## What was executed

The archived QGPF runs used previously inspected `runs-main-30` trajectories.
They therefore cannot be counted as external validation. This run adds a
separate, frozen synthetic regime whose trajectories are generated at runtime
from new seeds (`10000..10029`) and are not read from the historical archive.
The transition matrix and observation function also differ from the historical
constant-velocity/range contract:

\[
F_{\mathrm{ext}}=\begin{bmatrix}1&0.8&0&0\\0&0.85&0&0\\0&0&1&0.6\\0&0&0&0.9\end{bmatrix},\qquad
h(x)=\sqrt{x_1^2+0.5x_3^2}+0.2\sin(0.1x_1).
\]

The frozen process standard deviations are `(2.0, 0.75, 1.25, 0.06)` and the
observation standard deviation is `2.0`. Particle count, bit length, generation
count, elite fraction, rotation schedule, and method contracts are inherited
unchanged from the stable revision. Each of the 30 trajectory seeds is paired
across `B-stable`, `Q-stable`, and `Q-stable-retain`; truth is retained only by
the evaluator and is never supplied to the filter.

This is explicitly a **distribution-shift synthetic stress test**, not a
real-world external-data validation and not a recovery of the source author's
implementation.

## Execution evidence

* 90/90 run manifests completed; no failed seed was removed.
* All 90 numerical guardrail files pass: finite estimates, normalized weights,
  counter totals, read-only post endpoint, and truth isolation.
* For every seed, the three methods share exactly one observation-input hash.
* A fixed-seed replay of seed 0 for all three methods reproduced all scientific
  arrays and all metrics except operational timers exactly.
* Summary SHA-256: `36da6a475473747bd9c149de763d1928fb0cd177eeb7f952618ff2cf1c97714e`;
  raw metrics CSV SHA-256:
  `7c2205afce0fe9d63cc515cca557b2158fb2cbf6d17d8d646df9244820f263ce`.
* Stable implementation SHA-256:
  `01d14e073da2b5ff4c10a5f9edaa51c7cfac24d458e12b8ad30de78d5c348dac`.

## Descriptive results

Primary outcome is pre-move position RMSE over `k=1..100` and the two position
coordinates. Values below are trajectory-level summaries (`n=30`):

| Method | Mean RMSE | SD | Median | IQR | Mean velocity RMSE | Mean ESS | Mean distinct |
|---|---:|---:|---:|---:|---:|---:|---:|
| B-stable | 8.820 | 4.796 | 7.061 | 5.163–11.235 | 0.857 | 110.607 | 119.007 |
| Q-stable | 48.779 | 17.197 | 43.976 | 36.627–57.170 | 7.232 | 15.382 | 199.898 |
| Q-stable-retain | 48.819 | 17.249 | 45.965 | 38.245–54.119 | 6.945 | 15.908 | 199.907 |

Paired differences are method RMSE minus B-stable RMSE. The 20,000-draw paired
bootstrap intervals are exploratory descriptive intervals, not a claim of
equivalence or universal superiority:

| Contrast | Mean difference | 95% bootstrap CI | \(d_z\) | Lower-RMSE wins |
|---|---:|---:|---:|---:|
| Q-stable − B-stable | 39.959 | 33.613–47.016 | 2.089 | 0/30 |
| Q-stable-retain − B-stable | 39.999 | 33.782–47.081 | 2.113 | 0/30 |

The result is a stress-test observation: under this shifted regime the audited
QGPF-like contracts are substantially worse than the bootstrap comparator and
have much lower ESS. It does not identify whether the cause is encoding range,
proposal geometry, the changed observation map, or another contract choice.
The stable QGPF branches still encountered zero-valued probability-domain
diagnostics (mean candidate-zero counts 58,714 per 30-seed batch and
mean hypothetical zero-zero ties 15,830 for `Q-stable`); these are retained as
diagnostics and did not enter the stable log-domain acceptance decision.

## Files

* Frozen protocol: `protocol.json`.
* Runner: `run_external_validation.py`.
* Raw run bundles: `runs-external-30/`.
* Fixed-seed replay bundles: `runs-external-replay-seed0/`.
* Machine-readable integrity checks: `integrity_verification.json` and
  `replay_verification.json`.
* Machine-readable summary: `runs-external-30/summary.json`.

## Manuscript boundary

The main manuscript may use this result only as evidence that performance does
not automatically transport to a changed synthetic regime. It must not be
described as real external-data validation, a source-algorithm result, a quantum
hardware result, or proof of general QGPF failure. A true external-data claim
still requires a public dataset with a frozen test split and trajectory/entity
level ground truth.
