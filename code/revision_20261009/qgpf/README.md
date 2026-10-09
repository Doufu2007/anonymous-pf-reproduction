# Isolated QGPF revision (2026-10-09)

This directory implements the frozen local exploratory revision. Historical
code and runs in code/qgpf_reproduction/ are read-only. No published numerical
table or unknown filtering target is claimed to have been recovered.

## Scientific contract

States are ordered [x, vx, y, vy]. The primary endpoint is coordinate
position RMSE: square root of the mean squared x/y error over steps 1–100.
Velocity RMSE is reported separately. Seeds 0–29 are independent paired
units; time steps are not independent samples.

- B-stable: direct Gaussian log scores and subtract-max normalization for
  the historical bootstrap implementation.
- Q-stable: the same direct log scores, stable ratio acceptance
  log(U) < log L(candidate) - log L(current) for lower-score candidates,
  with the historical proposal, bit observation, elite, bounds, schedule,
  prior and estimation/weight-target conventions retained.
- Q-stable-retain: only the candidate retention rule changes to strict
  log-score improvement. It starts from exactly the same stream as Q-stable:
  seed + 2*1000003. Its changed acceptance branch naturally changes later
  RNG consumption; a different initial stream is not introduced.

All scored candidates and old states have finite Gaussian log scores.
Underflow of an absolute probability does not imply an unsupported state.
Invalid states, observations, weights or unsupported normalization stop a
run and produce structured failure evidence; no seed is excluded.

The primary estimate is read immediately after pre-move weighting. The post
estimate is an exploratory reading of the existing returned particles and
weights. Recording it adds no likelihood calls or RNG draws and does not
feed back into the algorithm. Truth is only available to the driver/evaluator,
not to the filtering module.

## Execution and evidence

Python 3.13 / NumPy 2.4.2 / SciPy 1.17.1 were used on local CPU. Tests use
stdlib unittest and fixed-seed generated properties; pytest and Hypothesis
were unavailable. Red and green evidence are retained. The full run records
exact source, input, frozen-protocol and test-log SHA-256 identities, CPU,
thread settings, Git state, initial/final particles, all pre/post particles
and weights, predictions, per-step counters, diagnostics, final RNG state
and operational logs.

    $env:PYTHONHASHSEED='0'
    python -m unittest test_stable.py -v
    python run_revision.py --out-dir runs-m1 --methods B-stable,Q-stable --seeds 0:29 --workers 4 --replay
    python analyze_revision.py

Execute C only after completion of the intervening EC stage:

    python run_revision.py --out-dir runs-s1 --methods Q-stable-retain --seeds 0:29 --workers 4 --replay
    python analyze_revision.py --include-retain

Runs are immutable: the driver refuses to overwrite an existing seed/method
directory. All output is constrained to this revision directory. Inputs are
checked against the historical freeze before every run. BLAS and OpenMP
use one thread per independent worker, with at most four workers. Parallel
runtime includes initialization, filtering and diagnostics but excludes I/O;
it is operational logging and does not support a speed comparison. Separate
serial single-load samples, when produced, must be labelled as such.

The fixed-seed replay uses a new process pool after the main batch. Its
comparison excludes timers, timestamps and process metadata, and checks 15
scientific arrays, all remaining metrics and final RNG state exactly. This
establishes same-environment repeatability, not cross-hardware identity.

## Counting units

Q variants at this fixed configuration score 120,000 state log likelihoods:
20,000 base + 16,000 initial-old + 64,000 candidate + 20,000 updated.
The historical proposal_attempts unit is 16,000 complete chromosome events;
the generation-level candidate attempts are 64,000. Candidate and fitness
labels refer to the same likelihood work and must not be added. Q variants
record 5,120,000 scalar Bernoulli bit observations, two per bit per candidate.
Conditional rotation calls and probability-entry updates are the same event,
also not additive. Explicit crossover is absent and exactly zero. These are
classical CPU program counts, not physical quantum gates or measurements.

Probability-domain underflow/tie diagnostics are evaluated on the new stable
trajectory. Stable-versus-floor-rule disagreements compare the same two
scores and the same acceptance uniform on that trajectory. They do not
measure a pathwise disagreement with the historical run.

## Analysis and boundaries

All endpoints are reconstructed from saved arrays and matched exactly to
run metrics. Weight normalization, endpoint reconstruction, step-counter
sums, fixed counts and manifest file hashes are checked. Analysis reports
mean/SD and median/IQR, paired t and seed-level percentile-bootstrap 95%
intervals (10,000 draws), and paired dz. The three frozen primary comparisons
form one Holm family: Q-stable minus Q-old, Q-stable minus B-stable, and
Q-stable-retain minus Q-stable. M1 output explicitly leaves Holm pending
until all three are available. Secondary and pre/post summaries are
exploratory descriptive readings without extra inferential claims.

The original local red log contains a local traceback path. Its anonymized
public copy and conversion hashes are separate artifacts; the original log
identity is preserved and it must be excluded from a public package.
