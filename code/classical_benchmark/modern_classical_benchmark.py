"""Executable modern-classical PF contracts used beside the quantum audits.

The implementations share the nonlinear/non-Gaussian state-space model in
``benchmark_pf.py``.  They are deliberately named *contracts*: the source
papers do not provide enough information to claim a reproduction of every
published implementation.  Each contract has a fixed parameter set and
reports raw per-seed metrics, operation counts and the exact configuration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


Q = 10.0
R = 1.0
T = 100
X0 = 0.1
P0 = 5.0
N = 100
SEEDS = tuple(range(30))


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


def normalise(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise FloatingPointError("non-finite or negative particle weight")
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / len(weights))
    return weights / total


def log_likelihood(z: float, particles: np.ndarray) -> np.ndarray:
    residual = z - observation(particles)
    return -0.5 * residual**2 / R


def ess(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(weights)))


def bounded_weight_contract(weights: np.ndarray, cap_factor: float = 2.0) -> np.ndarray:
    """Cap large weights and redistribute their excess.

    This is the explicitly declared bounded-weight (Chopthin-style) contract
    used here.  It preserves the particle locations and avoids silently
    claiming that undocumented implementation details of a particular
    Chopthin code base were recovered.
    """
    n = len(weights)
    cap = cap_factor / n
    w = np.asarray(weights, dtype=float).copy()
    w = normalise(w)
    for _ in range(4 * n):
        excess = float(np.sum(np.maximum(w - cap, 0.0)))
        if excess <= 1e-14:
            break
        over = w > cap
        under = ~over
        w[over] = cap
        capacity = np.maximum(cap - w[under], 0.0)
        capacity_total = float(np.sum(capacity))
        if capacity_total <= 1e-14:
            break
        w[under] += excess * capacity / capacity_total
    return normalise(w)


def run_contract(method: str, z: np.ndarray, rng: np.random.Generator, n: int = N) -> dict:
    particles = rng.normal(X0, math.sqrt(P0), size=n)
    weights = np.full(n, 1.0 / n)
    estimates = np.zeros(T + 1)
    ess_values: list[float] = []
    distinct_values: list[float] = []
    likelihood_evals = 0
    lookahead_evals = 0
    observation_evals = 0
    resampling_calls = 0
    start = time.perf_counter()

    for k in range(1, T + 1):
        particles = transition(particles, k) + rng.normal(0.0, math.sqrt(Q), size=n)
        current_logw = np.log(np.maximum(weights, np.finfo(float).tiny)) + log_likelihood(z[k], particles)
        likelihood_evals += n
        current_weights = normalise(np.exp(current_logw - np.max(current_logw)))

        if method == "bounded-weight-chopthin":
            weights = bounded_weight_contract(current_weights, cap_factor=2.0)
            estimates[k] = float(np.sum(weights * particles))
            ess_values.append(ess(weights))
            if ess(weights) < 0.50 * n:
                idx = systematic_resample(weights, rng)
                particles = particles[idx]
                weights = np.full(n, 1.0 / n)
                resampling_calls += 1
            distinct_values.append(float(np.unique(np.round(particles, 12)).size))
            continue

        if method == "enkf-pf-hybrid":
            # A particle-EnKF contract: weighted particle update followed by
            # one stochastic EnKF correction using the same observation.
            idx = systematic_resample(current_weights, rng)
            particles = particles[idx]
            weights = np.full(n, 1.0 / n)
            h = np.asarray(observation(particles), dtype=float)
            observation_evals += n
            x_mean = float(np.mean(particles))
            h_mean = float(np.mean(h))
            cov_xh = float(np.mean((particles - x_mean) * (h - h_mean)))
            var_h = float(np.var(h))
            gain = cov_xh / (var_h + R)
            perturbed_z = z[k] + rng.normal(0.0, math.sqrt(R), size=n)
            particles = particles + gain * (perturbed_z - h)
            estimates[k] = float(np.mean(particles))
            ess_values.append(float(n))
            distinct_values.append(float(np.unique(np.round(particles, 12)).size))
            resampling_calls += 1
            continue

        if method == "twisted-lookahead":
            # One-step lookahead twist.  This is an offline auxiliary-function
            # contract; it is not claimed to be a source-code reproduction.
            twisted_logw = current_logw.copy()
            if k < T:
                next_pred = transition(particles, k + 1)
                twisted_logw += 0.5 * log_likelihood(z[k + 1], next_pred)
                lookahead_evals += n
            weights = current_weights
            estimates[k] = float(np.sum(weights * particles))
            ess_values.append(ess(weights))
            twist_weights = normalise(np.exp(twisted_logw - np.max(twisted_logw)))
            idx = systematic_resample(twist_weights, rng)
            particles = particles[idx]
            weights = np.full(n, 1.0 / n)
            distinct_values.append(float(np.unique(np.round(particles, 12)).size))
            resampling_calls += 1
            continue

        if method == "soft-resampling-dpf-forward":
            # Forward contract used by differentiable PF implementations:
            # mixture proposal q = alpha*w + (1-alpha)/N and importance
            # correction w/q.  No gradient training is claimed here.
            alpha = 0.90
            weights = current_weights
            estimates[k] = float(np.sum(weights * particles))
            ess_values.append(ess(weights))
            proposal = alpha * weights + (1.0 - alpha) / n
            idx = systematic_resample(proposal, rng)
            particles = particles[idx]
            corrected = weights[idx] / np.maximum(proposal[idx], np.finfo(float).tiny)
            weights = normalise(corrected)
            distinct_values.append(float(np.unique(np.round(particles, 12)).size))
            resampling_calls += 1
            continue

        if method == "bootstrap-systematic":
            weights = current_weights
            estimates[k] = float(np.sum(weights * particles))
            ess_values.append(ess(weights))
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.full(n, 1.0 / n)
            distinct_values.append(float(np.unique(np.round(particles, 12)).size))
            resampling_calls += 1
            continue

        raise ValueError(f"unknown method: {method}")

    runtime = time.perf_counter() - start
    return {
        "method": method,
        "seed": int(rng.bit_generator._seed_seq.entropy),
        "n": int(n),
        "rmse": float(np.sqrt(np.mean((estimates[1:] - _truth_current[1:]) ** 2))),
        "mae": float(np.mean(np.abs(estimates[1:] - _truth_current[1:]))),
        "mean_ess": float(np.mean(ess_values)),
        "mean_ess_over_n": float(np.mean(ess_values) / n),
        "mean_distinct": float(np.mean(distinct_values)),
        "mean_distinct_over_n": float(np.mean(distinct_values) / n),
        "runtime_s": float(runtime),
        "likelihood_evals": int(likelihood_evals),
        "lookahead_evals": int(lookahead_evals),
        "observation_evals": int(observation_evals),
        "resampling_calls": int(resampling_calls),
    }


_truth_current: np.ndarray


def run_all(output_dir: Path) -> tuple[list[dict], dict]:
    global _truth_current
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = (
        "bootstrap-systematic",
        "bounded-weight-chopthin",
        "enkf-pf-hybrid",
        "twisted-lookahead",
        "soft-resampling-dpf-forward",
    )
    rows: list[dict] = []
    for seed in SEEDS:
        truth, observations = simulate_truth(np.random.default_rng(seed))
        _truth_current = truth
        for method_index, method in enumerate(methods):
            method_seed = seed + 100000 + 1000 * method_index
            result = run_contract(method, observations, np.random.default_rng(method_seed), N)
            result["seed"] = int(seed)
            result["method_seed"] = int(method_seed)
            rows.append(result)

    fieldnames = list(rows[0].keys())
    with (output_dir / "modern_classical_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, dict] = {}
    for method in methods:
        vals = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            "runs": len(vals),
            "n": N,
            "rmse_mean": float(np.mean([r["rmse"] for r in vals])),
            "rmse_sd": float(np.std([r["rmse"] for r in vals], ddof=1)),
            "mae_mean": float(np.mean([r["mae"] for r in vals])),
            "mean_ess": float(np.mean([r["mean_ess"] for r in vals])),
            "mean_ess_over_n": float(np.mean([r["mean_ess_over_n"] for r in vals])),
            "mean_distinct": float(np.mean([r["mean_distinct"] for r in vals])),
            "mean_distinct_over_n": float(np.mean([r["mean_distinct_over_n"] for r in vals])),
            "runtime_s_mean": float(np.mean([r["runtime_s"] for r in vals])),
            "likelihood_evals": sorted(set(r["likelihood_evals"] for r in vals)),
            "lookahead_evals": sorted(set(r["lookahead_evals"] for r in vals)),
            "observation_evals": sorted(set(r["observation_evals"] for r in vals)),
            "resampling_calls": sorted(set(r["resampling_calls"] for r in vals)),
        }
    config = {
        "model": {"Q": Q, "R": R, "T": T, "X0": X0, "P0": P0},
        "particle_count": N,
        "seeds": list(SEEDS),
        "methods": {
            "bootstrap-systematic": {"resampler": "systematic"},
            "bounded-weight-chopthin": {"cap_factor": 2.0, "resample_threshold": 0.50},
            "enkf-pf-hybrid": {"correction": "stochastic scalar EnKF after systematic PF resampling"},
            "twisted-lookahead": {"lookahead_steps": 1, "twist_exponent": 0.5, "offline": True},
            "soft-resampling-dpf-forward": {"alpha": 0.90, "gradient_training": False},
        },
        "contract_boundary": "These are declared classical contracts, not claims of exact source-code reproduction.",
    }
    (output_dir / "modern_classical_summary.json").write_text(
        json.dumps({"config": config, "aggregate": aggregate}, indent=2), encoding="utf-8"
    )
    (output_dir / "modern_classical_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    return rows, aggregate


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "modern_benchmark_results"
    _, aggregate = run_all(out)
    print(json.dumps(aggregate, indent=2))
