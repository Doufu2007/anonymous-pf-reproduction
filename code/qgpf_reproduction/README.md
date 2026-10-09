# QGPF-like independent audit

This directory contains a minimal, auditable independent implementation of the
quantum genetic particle filter (QGPF) described by Wei and Guo (2021),
“基于量子遗传粒子滤波的无线传感器网络的目标跟踪算法的研究”. It is an
independent implementation with documented source differences. No author code
was located. The result must not be called a strict or exact reproduction.

## Implemented source-level facts

- Four-dimensional constant-velocity state `[x, vx, y, vy]`.
- Range observation `sqrt(x^2+y^2)`.
- Binary quantum chromosome and quantum-rotation update.
- Particle weight as chromosome fitness and elite preservation.
- Position and velocity RMSE are reported separately.

## Explicit assumptions

The source specifies N=400, m=10, four generations, 99 simulations, MATLAB 7.2,
and retaining the original particle when a replacement has lower importance.
It reports an unlabelled ratio of 0.7; its relation to an elite fraction is
unresolved. The audit allows some lower-likelihood replacements and estimates
before genetic optimisation, whereas the source estimates afterwards.

The source paper does not specify the measurement-noise variance, exact process
noise convention, state encoding bounds, or a fully identified elite fraction.
The historical implementation records completion choices in
`runs-main-30/manifest.json`: observation standard deviation `1.0`, the reported
noise diagonal as additive standard deviations, 10 bits per state coordinate,
state bounds `[-1000,1000]` for positions and `[-100,100]` for velocities, 20%
elites, 4 generations, and a decreasing rotation angle. These choices are
frozen in `qgpf.py` and are intended for sensitivity analysis, not numerical
matching to Table 1. It also replaces the displayed source noise-input matrix
B (Eq. 13) with independent additive noise for the four state coordinates.

Source Section 4 explicitly lists amplitude initialisation, measurement,
fitness, rotation, and best retention. Crossover and mutation are described for
the classical GAPF in Section 2 and are not explicit QGPF steps; their absence
in this audit is not an established source deviation. Source Eq. 9 forms weights
from current sensor likelihoods. Weight-target consistency remains unresolved;
not carrying preceding nonuniform weights is not, by itself, a demonstrated
departure from that displayed source equation.

The original code and historical run manifests are preserved. The current
interpretation is recorded in `../../metadata/implementation_audit_2026-10-09.md`;
it corrects historical descriptions without changing the generated outputs.

## Run

```powershell
cd code/qgpf_reproduction
python run_qgpf.py --seeds 0:29 --out-dir runs-main-30
```

Outputs include a manifest, per-seed truth/observation/estimate arrays, raw
metrics, and summary. `bootstrap` is the matched classical PF baseline.

