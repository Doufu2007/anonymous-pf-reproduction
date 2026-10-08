"""Independent QM-HPF reimplementation with explicit assumptions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    horizon: int = 200
    initial_state: float = 1.0
    observation_variance: float = 1.0
    gamma_shape: float = 0.5
    gamma_base_scale: float = 0.5


@dataclass(frozen=True)
class FilterConfig:
    particle_count: int = 100
    initial_particle_std: float = 1.0
    generation_count: int = 20
    bit_length: int = 16
    state_min: float = -10.0
    state_max: float = 10.0
    rotation_angle_max_pi: float = 0.04
    rotation_angle_min_pi: float = 0.01


def transition(x: np.ndarray | float, k: int) -> np.ndarray | float:
    return 1.0 + np.sin(0.04 * np.pi * k) + 0.5 * x


def observation(x: np.ndarray | float, k: int) -> np.ndarray | float:
    return np.where(k <= 30, 0.2 * np.asarray(x) ** 2, 0.5 * np.asarray(x) - 2.0)


def simulate_truth(
    rng: np.random.Generator,
    model: ModelConfig,
    process: str = "gaussian",
    process_variance: float = 1.0,
    gamma_scale: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(model.horizon + 1, dtype=float)
    y = np.zeros(model.horizon + 1, dtype=float)
    x[0] = model.initial_state
    obs_std = math.sqrt(model.observation_variance)
    for k in range(1, model.horizon + 1):
        noise = rng.normal(0.0, math.sqrt(process_variance))
        if process == "gaussian_gamma":
            noise += gamma_scale * rng.gamma(model.gamma_shape, model.gamma_base_scale)
        x[k] = float(transition(x[k - 1], k)) + noise
        y[k] = float(observation(x[k], k)) + rng.normal(0.0, obs_std)
    return x, y


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def normalize_logweights(logw: np.ndarray) -> np.ndarray:
    shifted = logw - np.max(logw)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("non-finite particle weights")
    return weights / total


def particle_logweights(particles: np.ndarray, y: float, k: int, variance: float,
                        counter: dict[str, int] | None = None) -> np.ndarray:
    if counter is not None:
        counter["likelihood_evals"] += int(np.asarray(particles).size)
    residual = y - observation(particles, k)
    return -0.5 * residual**2 / variance


def ess(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights**2))


def encode_scalar(value: float, cfg: FilterConfig) -> np.ndarray:
    clipped = float(np.clip(value, cfg.state_min, cfg.state_max))
    scale = (2**cfg.bit_length - 1) / (cfg.state_max - cfg.state_min)
    integer = int(round((clipped - cfg.state_min) * scale))
    return np.array([(integer >> bit) & 1 for bit in range(cfg.bit_length)], dtype=np.int8)


def decode_scalar(bits: np.ndarray, cfg: FilterConfig) -> float:
    integer = int(sum(int(bit) << i for i, bit in enumerate(bits)))
    return float(cfg.state_min + integer * (cfg.state_max - cfg.state_min) / (2**cfg.bit_length - 1))


def rotate_bit_toward(
    current: int,
    target: int,
    rng: np.random.Generator,
    delta: float,
) -> int:
    """Measure a bit after rotating its amplitude toward the high-weight bit."""
    theta = delta if current != target else 0.0
    if current == 0 and target == 1:
        probability_one = math.sin(theta) ** 2
    elif current == 1 and target == 0:
        probability_one = math.cos(theta) ** 2
    else:
        probability_one = float(current)
    return int(rng.random() < probability_one)


def quantum_rotation_candidate(
    low_value: float,
    high_value: float,
    low_weight: float,
    high_weight: float,
    generation: int,
    cfg: FilterConfig,
    rng: np.random.Generator,
) -> float:
    low_bits = encode_scalar(low_value, cfg)
    high_bits = encode_scalar(high_value, cfg)
    if cfg.generation_count <= 1:
        fraction = 0.0
    else:
        fraction = generation / (cfg.generation_count - 1)
    delta = math.pi * (
        cfg.rotation_angle_max_pi * (1.0 - fraction)
        + cfg.rotation_angle_min_pi * fraction
    )
    # The paper's table uses the fitness ordering to select a rotation sign.
    # The measurement probability is represented directly, while the weight
    # ordering remains the acceptance criterion in Eq. (16).
    if high_weight <= low_weight:
        delta *= 0.5
    candidate_bits = np.array(
        [rotate_bit_toward(int(a), int(b), rng, delta) for a, b in zip(low_bits, high_bits)],
        dtype=np.int8,
    )
    return decode_scalar(candidate_bits, cfg)


def crossover_candidate(
    low_value: float,
    high_value: float,
    effective_sample_size: float,
    n: int,
    rng: np.random.Generator,
) -> float:
    alpha = rng.uniform(0.0, min(1.0, effective_sample_size / n))
    return float(alpha * low_value + (1.0 - alpha) * high_value)


def mh_candidate(
    low_value: float,
    high_value: float,
    low_weight: float,
    high_weight: float,
    rng: np.random.Generator,
    mutation_scale: float,
) -> float:
    del high_value, high_weight
    return float(low_value + rng.normal(0.0, mutation_scale))


def _accept_candidate(old_weight: float, candidate_weight: float, rng: np.random.Generator) -> bool:
    if old_weight <= 0:
        return candidate_weight > 0
    return bool(rng.random() < min(1.0, candidate_weight / old_weight))


def _resample_update(
    particles: np.ndarray,
    weights: np.ndarray,
    candidate_factory: Callable[[float, float, float, float, int], float],
    rng: np.random.Generator,
    cfg: FilterConfig,
    y: float,
    k: int,
    obs_variance: float,
    counter: dict[str, int],
    likelihood_budget: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(particles)
    order = np.argsort(-weights, kind="stable")
    threshold_index = min(n - 1, max(0, int(math.ceil(ess(weights))) - 1))
    threshold = float(weights[order[threshold_index]])
    high_mask = weights > threshold
    if not np.any(high_mask):
        high_mask[order[0]] = True
    low_indices = np.flatnonzero(~high_mask)
    high_indices = np.flatnonzero(high_mask)
    if len(low_indices) == 0:
        return particles.copy(), weights.copy()
    effective = ess(weights)
    updated = particles.copy()
    # ``weights`` are normalized posterior weights, while candidate/old
    # likelihoods below are unnormalized likelihoods. Keep all particles on
    # the normalized scale by applying only the likelihood ratio to a low
    # particle's original weight.
    updated_weights = weights.copy()
    for index in low_indices:
        # The original particle likelihood is needed to form a valid update.
        # Under a finite evaluation budget, leave this particle unchanged once
        # the budget cannot pay for that audit evaluation.
        if likelihood_budget is not None and counter["likelihood_evals"] + 1 > likelihood_budget:
            continue
        old_value = float(particles[index])
        old_weight = float(weights[index])
        candidate = old_value
        original_loglikelihood = float(
            particle_logweights(
                np.array([old_value]), y, k, obs_variance, counter
            )[0]
        )
        original_likelihood = float(np.exp(original_loglikelihood))
        old_likelihood = original_likelihood
        for generation in range(cfg.generation_count):
            # One generation evaluates the candidate and the current particle.
            # Stop before either call would exceed the declared budget.
            if likelihood_budget is not None and counter["likelihood_evals"] + 2 > likelihood_budget:
                counter["budget_stops"] = counter.get("budget_stops", 0) + 1
                break
            counter["proposal_attempts"] += 1
            high_index = int(rng.choice(high_indices))
            high_value = float(particles[high_index])
            high_weight = float(weights[high_index])
            candidate = candidate_factory(old_value, high_value, old_weight, high_weight, generation)
            counter["candidate_evals"] += 1
            candidate_logw = particle_logweights(
                np.array([candidate]), y, k, obs_variance, counter
            )[0]
            candidate_weight = float(np.exp(candidate_logw))
            counter["old_evals"] += 1
            old_logw = particle_logweights(
                np.array([old_value]), y, k, obs_variance, counter
            )[0]
            old_likelihood = float(np.exp(old_logw))
            if _accept_candidate(old_likelihood, candidate_weight, rng):
                counter["accepts"] += 1
                old_value = candidate
                old_likelihood = candidate_weight
        updated[index] = old_value
        likelihood_ratio = old_likelihood / max(original_likelihood, 1e-300)
        updated_weights[index] = weights[index] * likelihood_ratio
    updated_weights = updated_weights / np.sum(updated_weights)
    del effective
    indices = systematic_resample(updated_weights, rng)
    return updated[indices], np.full(n, 1.0 / n)


def run_filter(
    observations: np.ndarray,
    truth: np.ndarray,
    rng: np.random.Generator,
    method: str,
    model: ModelConfig = ModelConfig(),
    cfg: FilterConfig = FilterConfig(),
    process_variance: float = 1.0,
    likelihood_budget: int | None = None,
) -> dict[str, object]:
    particles = rng.normal(model.initial_state, cfg.initial_particle_std, size=cfg.particle_count)
    estimates = np.zeros(model.horizon + 1, dtype=float)
    ess_values: list[float] = []
    distinct_values: list[float] = []
    counter = {
        "likelihood_evals": 0,
        "candidate_evals": 0,
        "old_evals": 0,
        "proposal_attempts": 0,
        "accepts": 0,
        "budget_stops": 0,
    }
    start = time.perf_counter()
    for k in range(1, model.horizon + 1):
        if likelihood_budget is not None and counter["likelihood_evals"] + cfg.particle_count > likelihood_budget:
            raise RuntimeError(
                "likelihood budget exhausted before a complete filtering step; "
                "increase the budget or reduce the proposal schedule"
            )
        particles = transition(particles, k) + rng.normal(0.0, math.sqrt(process_variance), size=cfg.particle_count)
        weights = normalize_logweights(particle_logweights(
            particles, observations[k], k, model.observation_variance, counter
        ))
        estimates[k] = float(np.sum(weights * particles))
        ess_values.append(ess(weights))
        # Reserve the base likelihood evaluations required by all future
        # filtering steps. Proposal generations may spend only the remainder
        # of the current step's share of the global budget.
        step_likelihood_budget = (
            None
            if likelihood_budget is None
            else likelihood_budget - (model.horizon - k) * cfg.particle_count
        )
        if method == "bootstrap":
            particles = particles[systematic_resample(weights, rng)]
            weights = np.full(cfg.particle_count, 1.0 / cfg.particle_count)
        elif method == "mh":
            scale = 0.10 * max(float(np.std(particles)), 1e-8)
            particles, weights = _resample_update(
                particles, weights,
                lambda low, high, lw, hw, generation: mh_candidate(low, high, lw, hw, rng, scale),
                rng, cfg, observations[k], k, model.observation_variance, counter,
                step_likelihood_budget,
            )
        elif method in {"qm_hpf", "rotation_only", "crossover_only"}:
            effective = ess(weights)
            probability_rotation = 1.0 - effective / cfg.particle_count
            def factory(low: float, high: float, lw: float, hw: float, generation: int) -> float:
                if method == "rotation_only" or (method == "qm_hpf" and rng.random() < probability_rotation):
                    return quantum_rotation_candidate(low, high, lw, hw, generation, cfg, rng)
                return crossover_candidate(low, high, effective, cfg.particle_count, rng)
            if method == "crossover_only":
                factory = lambda low, high, lw, hw, generation: crossover_candidate(
                    low, high, effective, cfg.particle_count, rng
                )
            particles, weights = _resample_update(
                particles, weights, factory, rng, cfg, observations[k], k,
                model.observation_variance, counter, step_likelihood_budget,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        distinct_values.append(float(np.unique(np.round(particles, decimals=12)).size))
    error = estimates[1:] - truth[1:]
    return {
        "estimate": estimates,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_ess": float(np.mean(ess_values)),
        "mean_distinct": float(np.mean(distinct_values)),
        "runtime_s": float(time.perf_counter() - start),
        **counter,
    }
