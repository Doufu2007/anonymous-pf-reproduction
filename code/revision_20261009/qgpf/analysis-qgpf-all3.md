# QGPF numerical-revision evidence

Exploratory revision of an independent reimplementation on archived inputs. The primary is coordinate position RMSE; negative paired differences indicate smaller error. Parallel runtime is excluded from speed claims.

| Variant | n | Position RMSE mean ± SD | median [Q1, Q3] | Velocity RMSE mean ± SD |
|---|---:|---:|---:|---:|
| Q-old | 30 | 482.386352 ± 156.115333 | 406.726330 [376.540737, 588.506886] | 21.511354 ± 2.546386 |
| B-stable | 30 | 120.942440 ± 164.076898 | 74.648050 [32.328033, 105.400132] | 3.715032 ± 4.049697 |
| Q-stable | 30 | 489.551020 ± 123.734772 | 475.442839 [394.979664, 570.675926] | 22.665062 ± 2.606209 |
| Q-stable-retain | 30 | 491.909716 ± 151.895066 | 469.431073 [391.018782, 503.080999] | 22.988747 ± 2.564028 |

| Pair (left minus right) | Δ mean | paired t 95% CI | seed bootstrap 95% CI | dz | left smaller / 30 | Holm family3 p |
|---|---:|---|---|---:|---:|---:|
| Q-stable minus Q-old | 7.164668 | [-29.223780, 43.553115] | [-28.552374, 40.275089] | 0.073521 | 11 | 1 |
| Q-stable minus B-stable | 368.608580 | [287.244388, 449.972773] | [290.116456, 443.434984] | 1.691662 | 2 | 1.0851234e-09 |
| Q-stable-retain minus Q-stable | 2.358696 | [-36.663492, 41.380883] | [-32.771484, 41.017206] | 0.022571 | 17 | 1 |

COMPLETE_FAMILY3

All saved endpoints were reconstructed exactly from raw arrays. Run file hashes, finite estimates, normalized weights and per-step counter sums passed. Candidate/fitness/proposal scoring labels overlap likelihood scores; rotation/probability-update labels overlap. These are CPU program events, not hardware quantum gates.

No claim is made that stable ratios recover the source filtering target, unknown implementation choices, or the original published numerical table.
