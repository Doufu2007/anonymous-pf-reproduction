# Fixed-seed profiling supplement

This is a descriptive cost supplement, not a new accuracy experiment or an
equal-wall-clock comparison. It calls the scientific source functions unchanged:

- the three cached fair-budget methods, with their original 200-step model and
  at most 200 likelihood evaluations per step;
- QGPF-like, with the original 100-step model, 200 particles and four generations;
- the EC-QPF source selection module, with the original 20 one-step updates,
  500 particles and 500 swarm candidates.

The plan in `profile_frozen_plan.json` was saved before these measurements.
It declares seeds 0, 1 and 2, one ordinary run and one cProfile run per method
and seed, no warm-up and no outcome-driven configuration changes. Inputs and
algorithm RNG streams match the historical audit runners. Timing excludes input
generation, JSON output, hashing and imports; it includes initialization inside
the unchanged algorithm and construction of its configured RNG.

## Run

Use Python with NumPy and threadpoolctl installed. From the repository root:

```powershell
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:VECLIB_MAXIMUM_THREADS='1'
python code/audit_reporting/profile_runner.py --out-dir code/audit_reporting/profile-runs-new
python code/audit_reporting/profile_archive_check.py --profile-dir code/audit_reporting/profile-runs-new
```

The runner rejects existing output directories and never overwrites the
historical 30-seed results. NumPy threadpools are explicitly constrained to one
thread and their observed configuration is saved. CPU model, OS, Python, NumPy,
dtype and thread settings appear in `profile_environment.json`, without personal
usernames or executable/library absolute paths. This does not pin CPU frequency
or affinity and does not isolate the process from the OS scheduler.

## What the percentages mean

Each cProfile function's **exclusive** time is assigned once to a declared
bucket: likelihood; rotation/crossover/encoding; separately named measurement;
probability/fitness; visible RNG functions; named sorting/resampling; or other.
The percentages use the sum of exclusive times as the denominator and sum to
100% for every run. They do not add cumulative times, which would double-count
nested calls. The default cProfile timer measures elapsed durations in this CPU
execution, not hardware cycles or an exact process-CPU-time breakdown.

Only functions with explicit, interpretable names are assigned to a scientific
operation. Shared NumPy internals and mixed-purpose loop self time remain
`other`. This conservative attribution is intentionally incomplete as an
algorithm-phase decomposition. It must not be presented as the total cost of
likelihood, rotation, measurement or RNG.

In particular:

- QM-HPF-like `rotate_bit_toward` jointly rotates and samples a bit. Its exclusive
  time belongs to the combined rotation/encoding bucket; separate measurement
  time is **unavailable**.
- QGPF-like performs measurement, selection and acceptance inside
  `qgpf_resample`. Their independent times are **unavailable** and their mixed
  loop self time is reported through `other` and `top_functions.json`.
- EC `observe_binary` is a separately named measurement function, so its own
  exclusive time can be reported. Caller RNG time is not included in that helper.
- This NumPy/Cython build does not expose Generator `random`, `normal`, `choice`
  and `uniform` as independently timed cProfile functions. Separate RNG time is
  therefore **unavailable**, not zero. The numeric zero in the raw visible-RNG
  accumulator means no separately visible matching entry, not absence of RNG.
- Probability/fitness helpers are attributable, but normalization, reductions
  and inline acceptance logic are not a complete phase measurement. A displayed
  helper-only share must keep that qualifier.

Use `profile_measurement_limits.json` for display labels. A displayed dash or
`unavailable` must replace a raw 0.0% for an operation that was executed but
could not be independently observed. Classical methods have no quantum
measurement; their measurement field is `not applicable`.

## Evidence and checks

Every workload saves ordinary and profiled scientific outputs, configuration,
input identity hashes, source hashes, a sanitized raw `.prof`, all exclusive
function times, top functions, and ordinary/profiled elapsed time. Scientific
output comparison excludes `runtime_s` and its derived `ess_per_second` and
includes arrays, counters, fitness histories and summary values. A mismatch
terminates execution and preserves the failed workload.

`profile_archive_check.py` independently compares each ordinary replay to the
existing 30-seed archives: fair-budget estimates, final particles, operation-count
arrays and scientific metrics; QGPF estimates and metrics; EC source estimates,
selected indices and scientific operator counts. It reports both exact equality
and a declared numerical tolerance, without rewriting those archives.

`profile_seed_summary.csv` retains each seed separately. `profile_summary.json`
reports the arithmetic mean of per-seed exclusive percentages, not a pooled
time-weighted percentage. Profiling overhead is reported separately as
`(profiled wall / ordinary wall - 1) * 100`; it is not subtracted to invent
uninstrumented phase times. Three seeds with one timing each are insufficient
for precise timing inference or hardware-independent conclusions. These times
must not replace the original 30-seed experiment timings.

The relevant scientific conclusion is limited: equal likelihood-evaluation
budgets do not guarantee equal wall-clock costs. No equal-wall-clock experiment,
GPU run, physical quantum execution or quantum acceleration is claimed.

The retained `profile_findings.md` and `profile_excerpt.tex` contain the measured
descriptive summary for the delivered run bundle.
