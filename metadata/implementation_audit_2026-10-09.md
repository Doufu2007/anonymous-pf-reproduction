# Updated source-to-code interpretation

Checked on 9 October 2026 against the source full texts, current code, and
archived configurations. This replaces the interpretation in the 8 October
audit; it does not modify the historical implementations or outputs.

## QM-HPF-like

Source DOI: 10.13195/j.kzyjc.2025.1074. Source locators: Eqs. (6), (8)--(11),
(16), (21)--(22), and the algorithm table.

- Scalar model, N=100, T=200, x0=1 and the Gaussian process-variance-1 case are
  source-derived. Normal(1,1) initial particles are an audit assumption.
- The source describes persistent amplitudes and weight--amplitude mapping.
  Native v4 instead rebuilds deterministic 16-bit state encodings on [-10,10].
- The source generation loop encloses the particle loop. Native v4 instead
  encloses proposals inside each low-particle loop. The supplementary budget
  runner visits low particles round-robin, capped at **20 attempts per particle**
  and 100 candidate evaluations per time step. The prior audit's G=2 budget
  description is obsolete and does not describe the executed fair-budget-30 run.
- Source Eq. (16) is a weight-ratio rule. Observation-likelihood-only acceptance
  and final/original likelihood reweighting are executable audit conventions;
  they do not establish posterior invariance.
- The source algorithm estimates after the update; both reported audits record
  estimates before the move. Systematic final resampling is an added choice.
- The historical likelihood-only Gaussian mutation heuristic is not the
  supplementary conditional resample-move baseline. The latter jointly resamples
  ancestor/state and uses the transition-times-likelihood acceptance ratio.
- The executed supplement compares a common 40,000-call upper cap. QM-HPF-like
  uses fewer calls in some steps. It is not an exact equal-realised-cost test.

## QGPF-like

Source DOI: 10.16526/j.cnki.11-4762/tp.2021.04.050. Source locators: Section 2
(classical GAPF), Section 4 (QGPF steps, Eqs. (9)--(11)), Section 5 (Eq. (13),
simulation settings and replacement-retention prose).

| Item | Explicit source record | Actual audit / interpretation |
|---|---|---|
| N, encoding, generations | N=400, m=10, four generations | N=200, 10 bits per state coordinate, G=4; coordinate decoding added |
| Repeats/software | 99 simulations, MATLAB 7.2 | 30 seeds, Python/NumPy |
| Ratio | An unlabelled simulation ratio of 0.7 | Elite fraction 0.2 is added; no claim that source ratio means elite fraction |
| Replacement | Retain original when new importance is lower | Some lower-likelihood replacements allowed by a likelihood-ratio probability; clear departure |
| Estimate timing | Genetic optimisation followed by Eq. (11) estimate | Pre-optimisation estimate; clear departure |
| Noise input | Eq. (13) displays a 4-by-2 B matrix; the noise-diagonal convention is unclear | Independent additive four-coordinate noise; matrix replaced and diagonal interpreted as standard deviations |
| Genetic operators | Section 4 explicitly lists amplitude initialisation, measurement, fitness, rotation, and best retention | Probability rotations/measurements; no explicit crossover/mutation. These operators are described for GAPF in Section 2, so their absence is not an established QGPF departure |
| Weights | Eq. (9) forms a product of current sensor likelihoods | Current likelihood weights without preceding nonuniform weights; source filtering-target consistency is unresolved, not a proved source disagreement |
| Remaining parameters | Observation variance, encoding bounds, and a fully identified elite rule not recoverable | Standard deviation 1, positions +/-1000, velocities +/-100, elite fraction .2 and rotation .08pi to .01pi are declared completion choices |

No author code was located. Source uncertainties and explicit departures are
different categories. Poor QGPF-like performance does not test the original
algorithm faithfully.

## Source-aligned EC-QPF module

Source DOI: 10.1109/JSEN.2021.3113364. See
`code/source_aligned_audit/source_to_code_contract.json` and `frozen_plan.json`
in the supplementary repository for the separately executed one-step module.
It restores binary observation, persistent probability updates, generation
order, ellipse filtering, fitness and post-duplication estimation. It uses
controlled inputs and disclosed completion choices, not the complete source
IMU-positioning model. Its scores and costs are not likelihood-budget results
and must not be pooled with the sequential audits.
