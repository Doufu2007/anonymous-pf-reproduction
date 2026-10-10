# External synthetic-regime stress test

This directory contains a pre-declared, exploratory external-regime check for the
QGPF-like numerical audit.  It does not replace the historical source-aligned
contract and it does not recover the source author's code.  The check uses the
same stable implementation and the same three method contracts, but generates
new state/observation trajectories with a frozen dynamics/observation regime
that was not used by the archived `runs-main-30` inputs. This is a
**distribution-shift synthetic stress test**, not external real-world-data
validation and not a source-code reproduction.

The unit of analysis is one independently generated trajectory (30 seeds).  The
same trajectory and observations are passed to `B-stable`, `Q-stable`, and
`Q-stable-retain`; the filter never receives the truth.  Primary outcome is
pre-move position RMSE over all time points and the two position coordinates.
Velocity RMSE, ESS, distinct particles, and operation counters are secondary.

The external regime is intentionally fixed before execution:

* horizon = 100; initial state = `(100, -2, -45, 5)`;
* transition matrix is `[[1,.8,0,0],[0,.85,0,0],[0,0,1,.6],[0,0,0,.90]]`;
* observation is `sqrt(x^2 + .5 y^2) + .2 sin(.1 x)`;
* process standard deviations = `(2.0, 0.75, 1.25, 0.06)`;
* radial observation standard deviation = `2.0`;
* filter initial mean = `(95, -1.5, -40, 4.5)` and standard deviations
  = `(12, 2, 10, 2)`;
* particle count, 10-bit encoding, four generations, elite fraction and
  rotation schedule are unchanged from the stable revision;
* trajectory seeds are `10000 + seed` for `seed=0..29`; filter stream seeds are
  `2000003 + seed`, `3000006 + seed`, and `4000009 + seed` for the three methods.

No hyperparameters are selected from these results.  The script writes immutable
run directories, input hashes, code hashes, configuration, raw predictions,
metrics, and a paired summary.  Runtime values are operational logging only.

Run from the repository root:

```powershell
python code/revision_20261010/external_validation/run_external_validation.py
```
