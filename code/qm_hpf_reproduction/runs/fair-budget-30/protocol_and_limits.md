# Fair likelihood-cap supplementary experiment

This supplementary protocol was frozen before its new run, after the native-cost
QM-HPF-like and QGPF-like results had been inspected. It is exploratory and must
not be described as a preregistered confirmatory experiment.

The simulation uses the QM-HPF source-inspired one-dimensional model, 200 time
steps, initial truth 1, Gaussian process variance 1, and observation variance 1.
The observation is quadratic at steps 1--30 and linear thereafter. Particle
initialisation is Normal(1,1 variance). Seeds 0--29 generate independent truth
and observation sequences; each sequence is shared by all three methods.
Algorithm streams are independently offset by 1000003 times the method index.
The estimator is the weighted mean immediately after prediction and weighting,
before resampling or moving. RMSE is calculated over steps 1--200.

## Methods

- Bootstrap PF uses 200 particles and systematic resampling at every step. All
  200 likelihood calls are base evaluations.
- Fixed-ancestor resample-move uses 100 particles. After weighting, systematic
  resampling indexes the ancestor, current state, and cached likelihood together.
  Each resampled particle receives one symmetric Normal(0,0.1 squared) random-walk
  proposal. Its log acceptance ratio includes both transition density and
  observation likelihood. The ancestor stays fixed. Particles remain equally
  weighted after the move, with no second likelihood reweighting and no clipping.
  This kernel leaves the extended filtering target proportional to
  previous-ancestor-weight times transition density times observation likelihood
  invariant. One move is not a claim that a chain has converged; the kernel also
  does not preserve a finite empirical collection of point atoms.
- QM-HPF-like uses 100 particles and the deterministic binary encoding of the
  previous independent implementation. Each step permits at most 100 cached
  candidate evaluations. Low particles are visited in generation-major round
  robin order, at most 20 proposals each. The strict ESS-ranked weight threshold
  selects high and low sets. Proposals use quantum-inspired bit rotation or
  crossover, then likelihood-only acceptance and the legacy weight-times-
  likelihood-ratio convention. This asymmetric heuristic has no demonstrated
  posterior-invariance or general importance-correction guarantee. It is not
  author code and does not reconstruct persistent quantum amplitudes.

## Costs and statistics

The comparison imposes the same *upper cap* of 200 scalar likelihood evaluations
per step. Both classical comparators expend exactly 40000 evaluations per
trajectory. A low set with fewer than five particles cannot fill 100 candidate
slots without exceeding the 20-generation ceiling. No dummy calls or extra
high-particle moves are introduced. QM-HPF-like actual expenditure is therefore
38360--39220 per trajectory, mean 38831.333, and the result must not be called an
exact equal-cost comparison. Wall-clock, transition-density calculations, state
encoding, memory, and sorting costs remain different.

RMSE is the primary endpoint; MAE, pre-move ESS, post-move distinct-state count,
runtime, ESS per second, and actual operations are descriptive secondary
endpoints. Runtime includes particle initialisation and filtering, excludes
truth generation, testing, saving and statistical analysis. Each of two contrasts
is QM-HPF-like minus a classical comparator. Report paired mean RMSE difference,
two-sided Student-t 95% CI, paired Cohen dz, and a 10000-draw percentile bootstrap
over trajectory seeds. Holm correction covers the two paired t tests. No seeds
or outliers are excluded. Neither confidence interval excludes zero; this is
absence of observed advantage in this contract, not proof of equivalence.

Raw ESS and unique-state count depend on particle count. For cross-N inspection,
mean ESS/N is 0.794366 for Bootstrap, 0.795620 for resample-move, and 0.798631 for
QM-HPF-like. Distinct fractions are 0.794940, 0.999347 and 0.859995, respectively.
Do not use the larger absolute Bootstrap ESS as evidence of better normalized
weight balance.

## Verification boundary

Six tests cover an artificial gold log MH ratio and its reverse, agreement with
the analytic Gaussian conditional target in the linear observation regime,
ancestor-state-cache alignment, genuine low-set generation ceilings, budget and
weight invariants, and same-seed values excluding timing and timing-derived
ESS/second. The initial tests failed before the implementation existed. A later
test failure identified ESS/second as timing-dependent; it was correctly removed
from deterministic equality, while the estimate, metrics, counters and particles
remained checked. An independent seed-0 replay of all three methods reproduces
the saved estimates and non-timing metrics. Every saved trajectory RMSE was
recomputed and all 90 file manifests and per-step counters were verified.

These checks support same-code, same-environment reproducibility and the
specified mathematical target. They do not prove cross-hardware exact equality,
performance on other models, faithful source-code recovery, or quantum speedup.
