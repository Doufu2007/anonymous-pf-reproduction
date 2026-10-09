# Descriptive profiling findings

Measured on AMD Ryzen 9 7940H w/ Radeon 780M Graphics, Windows-11-10.0.26200-SP0, Python 3.13.7,
NumPy 2.4.2, float64 CPU, OpenBLAS single thread. No GPU was used.
The frozen supplement retained seeds 0, 1 and 2, with one ordinary and one
cProfile run per method/seed. All 15 scientific-output identity checks passed.
The corresponding archived scientific outputs also match exactly for all 15
workloads (`profile_archive_replay_check.json`: PASS, all bitwise
equal). Timing is deliberately excluded from scientific identity.

| Workload | Ordinary mean elapsed s | Profiled mean elapsed s | Mean profiler overhead | Two largest function-exclusive shares |
|---|---:|---:|---:|---|
| Bootstrap (200) | 0.03521 | 0.04454 | 31.30% | run_budget_filter (14.99%), <method 'reduce' of 'numpy.ufunc' objects> (8.67%) |
| Resample-move (100) | 0.03807 | 0.05370 | 41.57% | run_budget_filter (10.22%), resample_move (8.44%) |
| QM-HPF-like (100) | 0.99325 | 1.32752 | 34.14% | qm_budget_update (23.93%), observation (12.67%) |
| QGPF-like (200) | 25.07502 | 38.57295 | 53.87% | qgpf_resample (32.01%), _clip (13.25%) |
| EC source module | 1.99921 | 2.03292 | 2.30% | update_swarm (53.53%), source_select (23.50%) |

The top two rows are **function self times**, not inclusive phase times. The
complete seven-bucket partition and top20 functions are in each raw run and the
summary. QGPF-like's `other` share is 78.02%; its mixed-purpose `qgpf_resample`
self time is 32.01% and generic NumPy `_clip` self time is 13.25%. These costs
cannot be apportioned into independent rotation, measurement, RNG and acceptance
phases from this profile without guessing. QM-HPF-like's
`rotate_bit_toward` accounts for 6.96% as a joint rotation-and-measure helper,
already included in its rotation/encoding bucket, so do not add it again.
EC `update_swarm` and `subset_logfitness` account for 53.53% and 10.69%, while
`observe_binary` has a separately attributable 1.13% helper-only self time.

Independent RNG time is unavailable for this NumPy/Cython build. Independent
QM/QGPF measurement time is also unavailable. Their visible-RNG or measurement
accumulator zeros must not be displayed as absence of executed operations.
Shared NumPy helpers and mixed loops remain in `other`; this is a conservative
function-label partition, not a complete operation-cost model. The elapsed
timings include configured algorithm/RNG initialization and diagnostics but
exclude data generation and I/O. They do not replace the original 30-seed
experiment runtime values and do not constitute an equal-wall-clock experiment.

The fair-budget Bootstrap and resample-move methods use 40,000 likelihood
evaluations per run; QM-HPF-like can leave the cap partly unused. Nevertheless,
equal likelihood-evaluation budgets do not imply equal wall-clock costs.
Profile overhead varies markedly and is reported separately; profile shares
are not de-biased to fabricate uninstrumented phase times. Three seeds with one
timing per workload provide descriptive implementation evidence only.
