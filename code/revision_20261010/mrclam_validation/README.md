# MR.CLAM validation package

This directory implements a declared, reproducible transfer audit on the UTIAS MR.CLAM public dataset. It is not a complete reproduction of any source QPF paper. The Q variants are labelled `Q-stable` and `Q-stable-retain` because they transfer a quantum-inspired representation into the stated 3-D localization contract; their weights are not propagated as a standard Bayesian posterior across events.

## Data handling

Use the official UTIAS ASRL page (`https://asrl.utias.utoronto.ca/datasets/mrclam/`) and keep the downloaded archives outside this repository. The official page does not state a standard open-source licence, so raw archives, truth arrays and observation arrays are not redistributed. The fetch script records archive sizes, CRC/SHA-256 and an official-page snapshot. The analysis package publishes code, the frozen protocol, source hashes, aggregate metrics and figures only.

## Frozen audit

Robot 1 from datasets 8 and 9 is evaluated for the first 1200 seconds jointly supported by odometry, mapped observations and ground-truth timestamps. Ground-truth pose values are evaluator-only. Observations are spaced by at least one second; all observations at a retained timestamp are kept. B-stable, RM-stable, Q-stable and Q-stable-retain receive the same 1200 complete batch-likelihood evaluation budget per event. Thirty seeds quantify Monte Carlo variation on each fixed sequence, not independent real trajectories.

Dataset 8 retains 459 event batches and has 605 raw rows with unresolved barcodes; its archive has 14 mapped landmarks and a maximum retained-event gap of 244.29 s. Dataset 9 retains 975 event batches, has one stable-sorted odometry inversion and one unresolved barcode row outside the selected window. These ingest facts are recorded in the run metadata.

The 240 archived runs are analysed with `analyze_results.py`. The analysis validates all manifests, file hashes and recomputed endpoint metrics. A separate seed-0 replay was run for all 8 dataset/method units; each unit matched 15 scientific arrays and every non-timing metric exactly. Runtime is excluded from replay equality.

## Commands

```powershell
python run_mrclam.py --raw-dir <path-to-local-data/mrclam> --datasets 8 9 --seeds 0 1 ... 29 --out <output-dir> --workers 2
python analyze_results.py --runs <dataset-8-run-dir> <dataset-9-run-dir> --expected-runs 240 --out-dir <analysis-dir>
```

The archived result directories and raw input remain local. Do not upload third-party archives or Vicon truth arrays to the public repository.
