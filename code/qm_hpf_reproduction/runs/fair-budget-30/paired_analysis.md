# Fair likelihood-cap exploratory supplement

exploratory supplement designed after viewing native-cost results

| Method | N | RMSE mean +/- SD | MAE mean | ESS | Distinct | Runtime seconds | Likelihood calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| bootstrap_200 | 200 | 0.932882 +/- 0.063776 | 0.739577 | 158.873 | 158.988 | 0.0388 | 40000 |
| resample_move_100 | 100 | 0.938317 +/- 0.061252 | 0.744803 | 79.562 | 99.935 | 0.0495 | 40000 |
| qm_hpf_like_100 | 100 | 0.934915 +/- 0.062175 | 0.742077 | 79.863 | 85.999 | 1.2208 | 38831 |

Negative paired differences favour QM-HPF-like.

| Contrast | Mean difference | t 95% CI | Bootstrap 95% CI | dz | p | Holm p | Wins |
|---|---:|---|---|---:|---:|---:|---:|
| qm_hpf_like_100_minus_bootstrap_200 | +0.002033 | [-0.002826, +0.006891] | [-0.002549, +0.006783] | +0.1562 | 0.39919 | 0.40566 | 13/30 |
| qm_hpf_like_100_minus_resample_move_100 | -0.003402 | [-0.008743, +0.001938] | [-0.008289, +0.001588] | -0.2379 | 0.20283 | 0.40566 | 16/30 |

Limitations:

- One source-inspired Gaussian synthetic model; no cross-task generalization.
- Equal likelihood calls do not match wall-clock, particles, memory or transition-density operations.
- QM-HPF-like kernel is heuristic and not established posterior invariant; not author-code reproduction.
- Failure to reject a difference is not evidence of equivalence or noninferiority.
- The resample-move step size is fixed, untuned; one MH step does not imply convergence.
