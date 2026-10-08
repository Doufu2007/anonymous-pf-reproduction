# QM-HPF independent reimplementation

This directory contains an independent reimplementation of the QM-HPF method
described in Zhang et al., *Particle filter based on quantum Metropolis-Hastings
resampling for lithium-ion battery state-of-charge estimation* (DOI:
10.13195/j.kzyjc.2025.1074).

It is not the authors' source code. The paper omits several implementation
choices, so this project records them explicitly and reports whether the
published direction of effect can be reproduced under those assumptions.

## Frozen source facts

- Synthetic model: the one-dimensional model in equations (21)--(23).
- Horizon: `T=200`.
- Particle count in the source experiment: `N=100`.
- Initial state: `x0=1`.
- Observation-noise variance: `sigma_v^2=1`.
- Process-noise cases: Gaussian variances 1, 2, and 3; plus the paper's
  Gaussian--Gamma mixture cases.
- Reported baselines: AGPF and M-HPF.
- Reported metrics: RMSE and MAE.

## Explicit implementation assumptions

The source paper does not specify these values or procedures:

1. The QM-HPF generation count is set to `G=20`.
2. Continuous particles are encoded with a 16-bit fixed-point representation
   on `[-10, 10]`. The paper gives the quantum-bit update but not the encoding
   interval or `lenchrom`.
3. A low-weight particle is encoded as a deterministic bit state. Each bit is
   rotated toward the corresponding bit of a randomly selected high-weight
   particle by the paper's direction table and linearly decreasing angle.
4. The candidate is measured once per generation and decoded back to the fixed
   interval. The candidate is clipped to the interval before its likelihood is
   evaluated.
5. The non-Gaussian process component uses `Gamma(shape=0.5, scale=0.5)` and
   is multiplied by the source paper's `rho` value. After low-weight updates, systematic resampling is applied. The paper says
   that resampling is completed but does not name the final resampler.
6. The state transition, observation model, and likelihood use the direct
   scalar interpretation of equations (21)--(23).

These assumptions are part of the result. They must be changed only through a
new configuration, never silently.

## Run

```powershell
python .\run_reproduction.py --seeds 0:29 --out-dir .\runs-main-30-v4
python .\analyze_results.py .\runs-main-30-v4
python .\sensitivity.py --seeds 0:4 --out-dir .\runs-sensitivity-5
```

The command writes a configuration snapshot, environment record, per-seed raw
metrics, aggregate metrics, and a reproduction report. Use `pytest` to run the
invariants before running a larger sweep.

The earlier `main-30-v3` results were generated before the normalized-weight
scale correction and must not be used for the manuscript. The current
`main-30-v4` manifest records the corrected source-code hash.

## Interpretation labels

- `STRICT_REPRODUCTION`: author code and all omitted settings supplied.
- `NUMERICAL_REPRODUCTION`: independent implementation with complete settings
  that matches the source values within a predeclared tolerance.
- `QUALITATIVE_REPRODUCTION`: the proposed method has the same direction of
  effect as reported, but exact numerical matching is not established.
- `NOT_REPRODUCED`: the direction or reported ordering is not recovered.
