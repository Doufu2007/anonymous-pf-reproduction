"""Small reproducible benchmark for classical particle-filter baselines.

The model follows the nonlinear/non-Gaussian example used in the project
replication script.  It is deliberately a classical benchmark: it does not
claim to reproduce any quantum-inspired method.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np


Q = 10.0
R = 1.0
T = 100
X0 = 0.1
P0 = 5.0
SEEDS = tuple(range(30))
PARTICLE_COUNTS = (50, 100)


def transition(x: np.ndarray | float, k: int) -> np.ndarray | float:
    return x / 2.0 + 25.0 * x / (1.0 + x**2) + 8.0 * math.cos(1.2 * k)


def observation(x: np.ndarray | float) -> np.ndarray | float:
    return x**2 / 20.0


def simulate_truth(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(T + 1)
    z = np.zeros(T + 1)
    x[0] = X0
    for k in range(1, T + 1):
        x[k] = transition(x[k - 1], k) + rng.normal(0.0, math.sqrt(Q))
        z[k] = observation(x[k]) + rng.normal(0.0, math.sqrt(R))
    return x, z


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def stratified_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (np.arange(n) + rng.random(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def bootstrap_pf(z: np.ndarray, n: int, rng: np.random.Generator,
                 resampler: str = "systematic", regularize: bool = False) -> tuple[np.ndarray, float, float]:
    particles = rng.normal(X0, math.sqrt(P0), size=n)
    estimate = np.zeros(T + 1)
    ess_values = []
    distinct_values = []
    start = time.perf_counter()
    for k in range(1, T + 1):
        particles = transition(particles, k) + rng.normal(0.0, math.sqrt(Q), size=n)
        logw = -0.5 * (z[k] - observation(particles)) ** 2 / R
        logw -= np.max(logw)
        weights = np.exp(logw)
        weights /= np.sum(weights)
        estimate[k] = float(np.sum(weights * particles))
        ess_values.append(float(1.0 / np.sum(weights**2)))
        if resampler == "stratified":
            indices = stratified_resample(weights, rng)
        else:
            indices = systematic_resample(weights, rng)
        particles = particles[indices]
        if regularize:
            # A standard Gaussian jitter scale based on the post-resampling spread.
            h = 0.15 * max(float(np.std(particles)), 1e-8) * n ** (-1.0 / 5.0)
            particles = particles + rng.normal(0.0, h, size=n)
        distinct_values.append(float(np.unique(np.round(particles, decimals=12)).size))
    runtime = time.perf_counter() - start
    return estimate, float(np.mean(ess_values)), float(np.mean(distinct_values)), runtime


def ekf(z: np.ndarray) -> np.ndarray:
    estimate = np.zeros(T + 1)
    estimate[0] = X0
    covariance = P0
    for k in range(1, T + 1):
        previous = estimate[k - 1]
        predicted = transition(previous, k)
        derivative_f = 0.5 + 25.0 * (1.0 - previous**2) / (1.0 + previous**2) ** 2
        predicted_covariance = Q + derivative_f * covariance * derivative_f
        derivative_h = predicted / 10.0
        innovation_covariance = derivative_h * predicted_covariance * derivative_h + R
        gain = predicted_covariance * derivative_h / innovation_covariance
        estimate[k] = predicted + gain * (z[k] - observation(predicted))
        covariance = predicted_covariance - gain * derivative_h * predicted_covariance
    return estimate


def main() -> None:
    rows = []
    for seed in SEEDS:
        truth, observations = simulate_truth(np.random.default_rng(seed))
        ekf_estimate = ekf(observations)
        rows.append({
            "seed": seed,
            "n": 0,
            "method": "EKF",
            "rmse": float(np.sqrt(np.mean((ekf_estimate[1:] - truth[1:]) ** 2))),
            "mean_ess": "",
            "mean_distinct": "",
            "runtime_s": "",
        })
        for n in PARTICLE_COUNTS:
            for method, resampler, regularize in (
                ("Bootstrap-SysPF", "systematic", False),
                ("Bootstrap-StratPF", "stratified", False),
                ("Regularized-SysPF", "systematic", True),
            ):
                estimate, mean_ess, mean_distinct, runtime = bootstrap_pf(
                    observations, n, np.random.default_rng(seed + 100000 + n), resampler, regularize
                )
                rows.append({
                    "seed": seed,
                    "n": n,
                    "method": method,
                    "rmse": float(np.sqrt(np.mean((estimate[1:] - truth[1:]) ** 2))),
                    "mean_ess": mean_ess,
                    "mean_distinct": mean_distinct,
                    "runtime_s": runtime,
                })

    output_dir = Path(__file__).resolve().parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    with (output_dir / "classical_pf_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for row in rows:
        key = f"{row['method']}_N{row['n']}"
        summary.setdefault(key, []).append(row)
    aggregate = {
        key: {
            "runs": len(values),
            "rmse_mean": float(np.mean([v["rmse"] for v in values])),
            "rmse_sd": float(np.std([v["rmse"] for v in values], ddof=1)),
            "mean_ess": None if values[0]["mean_ess"] == "" else float(np.mean([v["mean_ess"] for v in values])),
            "mean_distinct": None if values[0]["mean_distinct"] == "" else float(np.mean([v["mean_distinct"] for v in values])),
            "runtime_s": None if values[0]["runtime_s"] == "" else float(np.mean([v["runtime_s"] for v in values])),
        }
        for key, values in summary.items()
    }
    (output_dir / "classical_pf_benchmark_summary.json").write_text(
        json.dumps({"model": {"Q": Q, "R": R, "T": T}, "seeds": list(SEEDS), "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
