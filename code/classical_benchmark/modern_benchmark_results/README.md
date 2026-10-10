# Modern classical PF contract benchmark

This directory contains the 30-seed benchmark requested for the focused review.
All methods use the same nonlinear/non-Gaussian model (`Q=10`, `R=1`, `T=100`),
the same generated truth/observation sequence within each seed, `N=100`, and
the same primary state RMSE definition.  The run is descriptive and does not
claim that these contracts reproduce undocumented source code line by line.

Methods:

* `bootstrap-systematic`: systematic bootstrap PF;
* `bounded-weight-chopthin`: capped-weight Chopthin-style contract with cap
  `2/N`, followed by systematic resampling only when transformed ESS is below
  `0.5N`;
* `enkf-pf-hybrid`: systematic PF resampling followed by one stochastic scalar
  EnKF correction;
* `twisted-lookahead`: one-step offline lookahead twist with exponent `0.5`;
* `soft-resampling-dpf-forward`: differentiable-PF forward contract using
  `q=0.9w+0.1/N` and importance correction. No gradient training is included.

`modern_classical_raw.csv` is the per-seed record. `modern_classical_summary.json`
contains the aggregate values, while `modern_classical_config.json` and
`environment.json` freeze the contract and execution environment. Re-run with:

```text
python modern_classical_benchmark.py
```

The lookahead contract uses the next simulated observation and is therefore an
offline auxiliary-function audit, not a causal online filter. The soft
resampling row tests the forward filtering pass only; it is not evidence that a
learned differentiable proposal improves accuracy.
