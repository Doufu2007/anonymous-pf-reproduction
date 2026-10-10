"""Config-driven one-dimensional QM-HPF source kernel.

This kernel implements the equations that are recoverable from Zhang et al.
(2025). Every source ambiguity is a required configuration field; there are no
scientific defaults in this module. The caller must keep the resulting trace
and contract hash before making any reproduction claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OneDConfig:
    horizon: int
    particle_count: int
    initial_state: float
    observation_variance: float
    initial_particle_std: float
    state_min: float
    state_max: float
    bit_length: int
    generation_count: int
    amplitude_persistence: bool
    final_systematic_resampling: bool
    rotation_angle_max_pi: float
    rotation_angle_min_pi: float
    low_order: str
    amplitude_mapping: str


def transition(x: np.ndarray, k: int) -> np.ndarray:
    return 1.0 + np.sin(0.04 * np.pi * k) + 0.5 * x


def observation(x: np.ndarray, k: int) -> np.ndarray:
    return np.where(k <= 30, 0.2 * np.asarray(x) ** 2, 0.5 * np.asarray(x) - 2.0)


def simulate_truth(rng: np.random.Generator, cfg: OneDConfig, process_variance: float) -> tuple[np.ndarray, np.ndarray]:
    truth = np.zeros(cfg.horizon + 1, dtype=np.float64)
    observations = np.zeros(cfg.horizon + 1, dtype=np.float64)
    truth[0] = cfg.initial_state
    for k in range(1, cfg.horizon + 1):
        truth[k] = transition(np.asarray([truth[k - 1]]), k)[0] + rng.normal(0.0, math.sqrt(process_variance))
        observations[k] = observation(np.asarray([truth[k]]), k)[0] + rng.normal(0.0, math.sqrt(cfg.observation_variance))
    return truth, observations


def _normalise(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("invalid source likelihood weights")
    return weights / total


def _likelihood(particles: np.ndarray, y: float, k: int, variance: float) -> np.ndarray:
    residual = y - observation(particles, k)
    return np.exp(-0.5 * residual * residual / variance) / math.sqrt(2.0 * math.pi * variance)


def _systematic(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    points = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, points, side="right")


def _encode(value: float, cfg: OneDConfig) -> np.ndarray:
    levels = (1 << cfg.bit_length) - 1
    clipped = float(np.clip(value, cfg.state_min, cfg.state_max))
    integer = int(round((clipped - cfg.state_min) / (cfg.state_max - cfg.state_min) * levels))
    return ((integer >> np.arange(cfg.bit_length)) & 1).astype(np.int8)


def _decode(bits: np.ndarray, cfg: OneDConfig) -> float:
    levels = (1 << cfg.bit_length) - 1
    integer = int(np.dot(bits.astype(np.int64), 1 << np.arange(cfg.bit_length)))
    return float(cfg.state_min + (cfg.state_max - cfg.state_min) * integer / levels)


def _rotate(prob_one: float, target: int, angle: float, sign: float) -> float:
    p = float(np.clip(prob_one, 0.0, 1.0))
    a0, a1 = math.sqrt(1.0 - p), math.sqrt(p)
    theta = sign * angle
    c, s = math.cos(theta), math.sin(theta)
    a0_new = c * a0 - s * a1
    a1_new = s * a0 + c * a1
    if target == 0:
        a0_new, a1_new = c * a0 + s * a1, -s * a0 + c * a1
    return float(np.clip(a1_new * a1_new, 0.0, 1.0))


def run_filter(
    observations: np.ndarray,
    rng: np.random.Generator,
    cfg: OneDConfig,
    process_variance: float,
    rotation_signs: dict[str, float],
) -> dict[str, Any]:
    """Run the configurable one-dimensional source kernel and return full traces."""
    if cfg.low_order not in {"stable_weight_descending", "source_order"}:
        raise ValueError("low_order must be resolved before strict execution")
    if cfg.amplitude_mapping not in {"weight_to_bit_one", "weight_to_bit_zero"}:
        raise ValueError("amplitude_mapping must be resolved before strict execution")
    n = cfg.particle_count
    particles = rng.normal(cfg.initial_state, cfg.initial_particle_std, size=n)
    weights = np.full(n, 1.0 / n)
    traces: list[dict[str, Any]] = []
    amplitudes: dict[int, np.ndarray] = {}
    estimates = np.zeros(cfg.horizon + 1, dtype=np.float64)

    for k in range(1, cfg.horizon + 1):
        particles = transition(particles, k) + rng.normal(0.0, math.sqrt(process_variance), size=n)
        likelihood = _likelihood(particles, float(observations[k]), k, cfg.observation_variance)
        weights = _normalise(np.log(np.maximum(likelihood, 1e-300)))
        effective = float(1.0 / np.sum(weights * weights))
        order = np.argsort(-weights, kind="stable") if cfg.low_order == "stable_weight_descending" else np.arange(n)
        threshold_index = min(n - 1, max(0, int(math.ceil(effective)) - 1))
        threshold = float(weights[order[threshold_index]])
        high = order[weights[order] > threshold]
        low = order[weights[order] <= threshold]
        if len(high) == 0:
            high = order[:1]
        updated_particles = particles.copy()
        updated_weights = weights.copy()
        step_trace: dict[str, Any] = {
            "time_index": k,
            "predicted_particles": particles.copy(),
            "pre_update_weights": weights.copy(),
            "effective_sample_size": effective,
            "high_weight_indices": high.copy(),
            "low_weight_indices": low.copy(),
            "candidate_particles": [],
            "candidate_weights": [],
            "acceptance_uniforms": [],
            "accepted_flags": [],
            "generations": [],
        }
        lam = 1.0 - effective / n
        for low_index in low:
            low_value = float(particles[low_index])
            low_weight = float(weights[low_index])
            current = low_value
            current_weight = low_weight
            bit_probs = amplitudes.get(int(low_index)) if cfg.amplitude_persistence else None
            if bit_probs is None:
                if cfg.amplitude_mapping == "weight_to_bit_one":
                    bit_probs = np.full(cfg.bit_length, low_weight, dtype=np.float64)
                else:
                    bit_probs = np.full(cfg.bit_length, 1.0 - low_weight, dtype=np.float64)
            for generation in range(cfg.generation_count):
                branch_u = float(rng.random())
                high_index = int(rng.choice(high))
                high_bits = _encode(float(particles[high_index]), cfg)
                low_bits = _encode(current, cfg)
                if branch_u < lam:
                    fraction = generation / max(1, cfg.generation_count - 1)
                    angle = math.pi * (cfg.rotation_angle_max_pi * (1.0 - fraction) + cfg.rotation_angle_min_pi * fraction)
                    bits = np.empty(cfg.bit_length, dtype=np.int8)
                    for bit in range(cfg.bit_length):
                        bits[bit] = int(rng.random() < bit_probs[bit])
                        if bits[bit] != high_bits[bit]:
                            key = f"{int(low_bits[bit])}{int(high_bits[bit])}"
                            if key not in rotation_signs:
                                raise ValueError(f"missing source rotation sign for pattern {key}")
                            bit_probs[bit] = _rotate(bit_probs[bit], int(high_bits[bit]), angle, float(rotation_signs[key]))
                    candidate = _decode(bits, cfg)
                    branch = "rotation"
                else:
                    cross_alpha = float(rng.uniform(0.0, min(1.0, effective / n)))
                    candidate = cross_alpha * current + (1.0 - cross_alpha) * float(particles[high_index])
                    branch = "crossover"
                    angle = None
                candidate_likelihood = float(_likelihood(np.asarray([candidate]), float(observations[k]), k, cfg.observation_variance)[0])
                old_likelihood = float(_likelihood(np.asarray([current]), float(observations[k]), k, cfg.observation_variance)[0])
                acceptance = min(1.0, candidate_likelihood / max(old_likelihood, 1e-300))
                acceptance_u = float(rng.random())
                accepted = acceptance_u < acceptance
                if accepted:
                    current, current_weight = candidate, candidate_likelihood
                step_trace["candidate_particles"].append({"index": int(low_index), "generation": generation + 1, "branch": branch, "value": candidate, "high_index": high_index, "rotation_angle": angle})
                step_trace["candidate_weights"].append(candidate_likelihood)
                step_trace["acceptance_uniforms"].append(acceptance_u)
                step_trace["accepted_flags"].append(accepted)
            amplitudes[int(low_index)] = bit_probs.copy()
            updated_particles[low_index] = current
            updated_weights[low_index] = current_weight
        updated_weights /= updated_weights.sum()
        estimates[k] = float(np.sum(updated_particles * updated_weights))
        if cfg.final_systematic_resampling:
            selected = _systematic(updated_weights, rng)
            particles = updated_particles[selected]
            weights = np.full(n, 1.0 / n)
        else:
            particles, weights = updated_particles, updated_weights
        step_trace["post_update_particles"] = particles.copy()
        step_trace["post_update_weights"] = weights.copy()
        step_trace["estimate"] = estimates[k]
        step_trace["rng_state_after_step"] = rng.bit_generator.state
        traces.append(step_trace)
    return {"estimate": estimates, "traces": traces, "final_particles": particles, "final_weights": weights}
