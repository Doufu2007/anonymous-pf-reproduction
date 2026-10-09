# EC q-zero interpretation analysis

Unit: 30 datasets, each aggregating 20 updates into one 2D vector RMSE.

| Method | Mean RMSE | SD | Min | Max |
|---|---:|---:|---:|---:|
| ec_qzero_consistent | 4.0318749 | 0.352549 | 3.279112 | 4.7045548 |
| source_ecqpf_operator | 4.5504583 | 0.39312057 | 3.6499822 | 5.3399904 |
| ellipse_exact_fitness | 1.614782 | 0.23415984 | 1.3043669 | 2.3782238 |
| ellipse_systematic_fitness | 4.7223624 | 0.40262581 | 3.8472399 | 5.5676163 |

| Comparison | Mean diff | Bootstrap 95% CI | Sign-flip p | Holm-3 p | dz |
|---|---:|---:|---:|---:|---:|---:|
| consistent_minus_literal | -0.51858346 | [-0.54733344, -0.49022472] | 1.99996e-05 | 5.99988e-05 | -6.374634901407766 |
| consistent_minus_exact | 2.4170928 | [2.2691638, 2.5476503] | 1.99996e-05 | 5.99988e-05 | 6.067393967420229 |
| consistent_minus_systematic | -0.69048754 | [-0.72301964, -0.65813986] | 1.99996e-05 | 5.99988e-05 | -7.462775039047022 |

Bootstrap resamples whole datasets; sign-flip Monte Carlo resolution is1/50001. Holm family is the three frozen comparisons.
Secondaries and operation counts are descriptive and retained in analysis.json. No secondary hypothesis tests were added.
Historical comparator runtime is retained only as provenance; this revision does not make a cross-batch speed claim.
Consistent is a self-consistent q-zero interpretation variant, not original author intent or a complete source tracker.
