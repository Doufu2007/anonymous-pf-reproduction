# EC-QPF source-aligned selection-operator audit

This supplementary audit implements the selection operators in Wan et al.,
*Error Constraint Enhanced Particle Filter Using Quantum Particle Swarm Optimization*,
IEEE Sensors Journal 21(21), 24431–24439 (2021), DOI `10.1109/JSEN.2021.3113364`.
It does **not** reproduce the original IMU tracking experiment or complete EC-QPF.
No publisher full text is redistributed here.

The local plan was frozen before these new runs, after earlier manuscript audits
already existed. It is an exploratory implementation audit, not an externally
registered or confirmatory study. See `frozen_plan.json` and
`source_to_code_contract.json` for every source locator and completion choice.

## What is implemented

- Covariance ellipse filtering, source Eqs. (17)–(21).
- Literal binary observation in Eq. (12).
- Nonempty-subset fitness in Eq. (22).
- Persistent probability vectors, personal/global best and Eqs. (23)–(25).
- Source coefficients `alpha=0.8`, `c1=c2=0.2`, and sizes `N=np=500`.
- Generation outside swarm evaluation, stopping and estimation after duplication.

The literal observation formula treats `q` as the probability of bit zero. The
source later describes it as the probability of selection. This unresolved
convention conflict is retained in the contract. The experiment implements the
literal equations and has not tested an alternative convention. This alone
precludes attributing its results to the authors' original implementation.

Unreported choices include `IterMax=20`, a 95% covariance ellipse, scalar fitness
variance as the mean coordinate variance, invalid empty subsets, uniform initial
probabilities and balanced cyclic duplication. The supplied ellipse center is an
observed noisy state, shared by all main comparators. Its construction differs
from the source IMU pipeline.

## Controlled input and fair comparison

For each of 30 independent seed datasets, 20 independent one-step updates draw
`x ~ N(0,25I)`, `z=x+N(0,I)` and 500 prior particles. Algorithms receive only the
particle cloud and `z`; truth is used exclusively by the evaluator. These are
state coordinates in a specified synthetic model, not the original meters-based
tracking scene. They do not form a sequential filter trajectory.

Both primary classical comparators use the identical ellipse and source fitness:

1. An exact nearest-singleton optimizer. The mean distance of any nonempty subset
   cannot be less than its minimum distance, so the nearest singleton exactly
   maximizes Eq. (22). It is a transparent classical optimization reference.
2. Systematic resampling with normalized singleton-fitness scores.

Full-cloud systematic resampling with the true Gaussian observation likelihood is
a contextual reference. It does not share the ellipse or fitness and is excluded
from the two primary paired comparisons. No method is tuned using these outputs.

## Results

Primary RMSE is the square root of the mean squared Euclidean state error over
20 one-step updates within each seed dataset; the table reports the mean and
sample SD over 30 such independent datasets.

| Completed module | State RMSE | Mean distinct returned particles | Wall seconds per 20 updates |
|---|---:|---:|---:|
| Literal EC-QPF selection operators | 4.5505 ± 0.3931 | 188.13 | 1.9595 |
| Same ellipse/fitness, exact singleton | 1.6148 ± 0.2342 | 1.00 | 0.0139 |
| Same ellipse/fitness, systematic | 4.7224 ± 0.4026 | 385.13 | 0.0112 |
| Full-cloud Gaussian-likelihood systematic (context) | 1.5054 ± 0.2354 | 45.59 | 0.0030 |

Source minus exact-comparator RMSE difference is `+2.93568`, paired 95% bootstrap
CI `[2.78723, 3.07206]`. Source minus same-fitness systematic difference is
`-0.171904`, CI `[-0.186814, -0.156738]`. Seed-dataset bootstrap used 20,000 draws;
50,000 sign flips gave Holm-adjusted permutation `p=0.0000400` for both comparisons.
These numbers describe the completed module and controlled input only.

The source module averaged 111,700 subset-fitness evaluations per dataset versus
7,717 singleton-fitness evaluations for either shared-constraint comparator.
They are not likelihood evaluations. Binary dimensions processed and candidate
distances are archived separately. Wall time includes module initialization and
selection but excludes synthetic-input generation and file writing; it is specific
to the recorded CPU environment. No runtime speedup on quantum hardware is tested.

The exact optimizer collapses to one distinct particle, so maximizing this
fitness is not itself a diversity criterion. The completed swarm and classical
comparators exhibit different error/diversity trade-offs. This is a bounded
mechanism comparison, not evidence that the original EC-QPF succeeds or fails.
Sampling-weight ESS is undefined for the binary source and exact selectors;
their selected-particle counts are not called ESS.

## Raw evidence and verification

- `runs-main-30/seed-00/` through `seed-29/`: input clouds, observations, evaluation
  truth, returned original-particle indices, estimates, update metrics, source
  generation histories, costs, environment and executed-code snapshots.
- `runs-main-30/seed_metrics.csv`, `analysis.json`: recomputed summaries and paired
  comparisons; all 600 planned updates completed, without seed replacements.
- `fixed-repro-final-a/seed-00/`, `fixed-repro-final-b/seed-00/`: independent processes with
  identical prediction and deterministic raw-metric hashes in this environment.
- `artifact_validation.json`: all 34 run bundles passed hash/artifact checks.
- `test-red.log`, `test-red-baseline.log`, `test-green.log`: nine meaningful source
  operator tests, preserving failure-before-implementation and passing evidence.
  Machine-local traceback paths were removed; `log_redaction_provenance.json`
  records original and public-log hashes.

The first runs omitted the dirty Git diff hash. Their manifests explicitly record
verification-time completion of that provenance field; executed code remains
archived by content hash and numerical outputs are unchanged. The current runner
captures the Git diff hash at run start and requires the declared hash seed before
launch; the final independent pair verifies this current runner. This correction does not retroactively
claim an execution-time diff capture for old runs.

## Re-run

Use the recorded NumPy/Python environment for the same-environment contract.
Choose a fresh output directory because existing runs are never overwritten.

```powershell
$env:PYTHONHASHSEED='0'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m unittest -v test_operator.py
python run_audit.py --out runs-rerun-30
```

`analyze_audit.py` recomputes the delivered `runs-main-30` dataset. Its reported
statistics do not establish cross-hardware reproducibility, posterior consistency,
full source tracking replication, or the efficacy of the original algorithm.

## ESS reporting clarification

No comparable pre-resampling importance-weight ESS is specified by the binary
selection objective, so it is not reported for the source module or exact
singleton. The post-duplication equal weights would trivially give ESS=N, even
if all copies are identical. Distinct-state counts remain separate; the archived
sampling-ESS null fields mean this diagnostic was not assessed, rather than a
claim that no mathematical ESS can be defined for equal-weight outputs.
