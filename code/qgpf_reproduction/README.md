# QGPF independent reproduction

This directory contains a minimal, auditable independent implementation of the
quantum genetic particle filter (QGPF) described by Wei and Guo (2021),
“基于量子遗传粒子滤波的无线传感器网络的目标跟踪算法的研究”. It is an
independent reimplementation; the source paper does not provide code and omits
several numerical choices, so the result must not be called a strict or exact
reproduction.

## Implemented source-level facts

- Four-dimensional constant-velocity state `[x, vx, y, vy]`.
- Range observation `sqrt(x^2+y^2)`.
- Binary quantum chromosome and quantum-rotation update.
- Particle weight as chromosome fitness and elite preservation.
- Position and velocity RMSE are reported separately.

## Explicit assumptions

The source paper does not specify the measurement-noise variance, exact process
noise interpretation, state encoding bounds, elite fraction, acceptance rule,
or all QGA stopping details. The implementation records these assumptions in
`runs-main-30/manifest.json`: observation standard deviation `1.0`, the reported
noise diagonal as additive standard deviations, 10 bits per state coordinate,
state bounds `[-1000,1000]` for positions and `[-100,100]` for velocities, 20%
elites, 4 generations, and a decreasing rotation angle. These choices are
frozen in `qgpf.py` and are intended for sensitivity analysis, not numerical
matching to Table 1.

## Run

```powershell
cd code/qgpf_reproduction
python run_qgpf.py --seeds 0:29 --out-dir runs-main-30
```

Outputs include a manifest, per-seed truth/observation/estimate arrays, raw
metrics, and summary. `bootstrap` is the matched classical PF baseline.

