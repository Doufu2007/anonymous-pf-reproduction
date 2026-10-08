# Paired analysis

Run directory: `code/qm_hpf_reproduction/runs-main-30-v4`

## Aggregate RMSE

| Method | Mean | SD | 95% CI |
|---|---:|---:|---:|
| bootstrap | 0.9342 | 0.0586 | [0.9123, 0.9561] |
| mh | 0.9356 | 0.0625 | [0.9122, 0.9589] |
| qm_hpf | 0.9342 | 0.0630 | [0.9107, 0.9578] |
| rotation_only | 0.9439 | 0.0652 | [0.9195, 0.9682] |
| crossover_only | 0.9368 | 0.0616 | [0.9138, 0.9598] |

## Paired RMSE differences

Negative values favour QM-HPF.

| Comparison | Mean difference | 95% CI | Bootstrap CI | Cohen dz | Wins | Paired t p | Wilcoxon p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qm_hpf_minus_bootstrap | 0.0000 | [-0.0061, 0.0061] | [-0.0059, 0.0056] | 0.000 | 16/30 | 0.9986 | 0.9193 | 0.9986 |
| qm_hpf_minus_mh | -0.0013 | [-0.0051, 0.0024] | [-0.0050, 0.0021] | -0.134 | 15/30 | 0.4683 | 0.6263 | 0.9367 |
| qm_hpf_minus_rotation_only | -0.0097 | [-0.0164, -0.0029] | [-0.0161, -0.0036] | -0.536 | 22/30 | 0.00642 | 0.004032 | 0.02568 |
| qm_hpf_minus_crossover_only | -0.0026 | [-0.0075, 0.0023] | [-0.0074, 0.0019] | -0.197 | 15/30 | 0.2888 | 0.5291 | 0.8664 |

The paired tests quantify uncertainty across the fixed seed set; they do not establish equivalence or hardware acceleration.
