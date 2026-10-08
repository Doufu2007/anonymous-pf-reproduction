"""Independent quantum-inspired genetic particle filter benchmark.

This module implements a conservative, explicitly parameterized QGPF-like
resampling experiment based on Wei & Guo (2021), *Computer Measurement &
Control*. It is an independent reimplementation, not author code recovery.
The source specifies the constant-velocity model, binary qubit chromosome,
quantum rotation update and elite preservation, but omits several numerical
choices. All omitted choices are recorded in the manifest by run_qgpf.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    horizon: int = 100
    initial_truth: Tuple[float, float, float, float] = (70.0, 5.0, 23.0, 8.0)
    process_std: Tuple[float, float, float, float] = (1.5, 1.5, 0.9, 0.03)
    observation_std: float = 1.0


@dataclass(frozen=True)
class FilterConfig:
    particle_count: int = 200
    initial_mean: Tuple[float, float, float, float] = (71.0, 8.66, 18.9, 7.8)
    initial_std: Tuple[float, float, float, float] = (8.0, 1.0, 5.0, 1.0)
    state_min: Tuple[float, float, float, float] = (-1000.0, -100.0, -1000.0, -100.0)
    state_max: Tuple[float, float, float, float] = (1000.0, 100.0, 1000.0, 100.0)
    bit_length: int = 10
    generation_count: int = 4
    elite_fraction: float = 0.20
    rotation_angle_max_pi: float = 0.08
    rotation_angle_min_pi: float = 0.01


F = np.array([[1.0, 1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 1.0],
              [0.0, 0.0, 0.0, 1.0]])


def transition(states: np.ndarray) -> np.ndarray:
    return states @ F.T


def observation(states: np.ndarray) -> np.ndarray:
    return np.sqrt(states[:, 0] ** 2 + states[:, 2] ** 2)


def simulate_truth(rng: np.random.Generator, model: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    truth = np.zeros((model.horizon + 1, 4), dtype=float)
    obs = np.zeros(model.horizon + 1, dtype=float)
    truth[0] = np.asarray(model.initial_truth, dtype=float)
    process_std = np.asarray(model.process_std, dtype=float)
    for k in range(1, model.horizon + 1):
        truth[k] = transition(truth[k - 1:k])[0] + rng.normal(0.0, process_std)
        obs[k] = float(np.sqrt(truth[k, 0] ** 2 + truth[k, 2] ** 2)
                       + rng.normal(0.0, model.observation_std))
    return truth, obs


def normalize_logweights(logw: np.ndarray) -> np.ndarray:
    shifted = logw - np.max(logw)
    w = np.exp(shifted)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("invalid particle weights")
    return w / total


def likelihood(particles: np.ndarray, y: float, model: ModelConfig,
               counter: dict[str, int] | None = None) -> np.ndarray:
    if counter is not None:
        counter["likelihood_evals"] += int(len(particles))
    residual = y - observation(particles)
    var = model.observation_std ** 2
    return np.exp(-0.5 * residual ** 2 / var) / math.sqrt(2.0 * math.pi * var)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    points = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, points, side="right")


def encode_state(state: np.ndarray, cfg: FilterConfig) -> np.ndarray:
    lo = np.asarray(cfg.state_min)
    hi = np.asarray(cfg.state_max)
    levels = (1 << cfg.bit_length) - 1
    ints = np.rint((np.clip(state, lo, hi) - lo) / (hi - lo) * levels).astype(np.int64)
    bits = np.zeros((len(state), cfg.bit_length), dtype=np.int8)
    for bit in range(cfg.bit_length):
        bits[:, bit] = (ints >> bit) & 1
    return bits


def decode_state(bits: np.ndarray, cfg: FilterConfig) -> np.ndarray:
    lo = np.asarray(cfg.state_min)
    hi = np.asarray(cfg.state_max)
    levels = (1 << cfg.bit_length) - 1
    powers = (1 << np.arange(cfg.bit_length, dtype=np.int64))
    ints = bits.astype(np.int64) @ powers
    return lo + (ints / levels) * (hi - lo)


def _rotate_probability(prob_one: float, target_bit: int, angle: float) -> float:
    """One real-amplitude quantum rotation, represented as a probability."""
    prob_one = float(np.clip(prob_one, 0.0, 1.0))
    c, s = math.cos(angle), math.sin(angle)
    a0, a1 = math.sqrt(max(0.0, 1.0 - prob_one)), math.sqrt(prob_one)
    if target_bit == 1:
        a0, a1 = c * a0 - s * a1, s * a0 + c * a1
    else:
        a0, a1 = c * a0 + s * a1, -s * a0 + c * a1
    return float(np.clip(a1 * a1, 0.0, 1.0))


def qgpf_resample(particles: np.ndarray, weights: np.ndarray, y: float,
                  rng: np.random.Generator, model: ModelConfig,
                  cfg: FilterConfig,
                  counter: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Quantum-inspired genetic resampling with explicit elite preservation."""
    n, dim = particles.shape
    elite_n = max(1, min(n - 1, int(round(cfg.elite_fraction * n))))
    order = np.argsort(-weights, kind="stable")
    elite_idx = order[:elite_n]
    elites = particles[elite_idx].copy()
    offspring = np.empty((n - elite_n, dim), dtype=float)
    elite_weights = weights[elite_idx]
    # Initial QGA chromosome amplitudes are |0>+|1> over sqrt(2), hence p(1)=0.5.
    levels = (1 << cfg.bit_length) - 1
    for out_i in range(n - elite_n):
        parent_i = int(rng.choice(elite_n, p=elite_weights / np.sum(elite_weights)))
        target = elites[parent_i]
        current = particles[order[elite_n + (out_i % (n - elite_n))]]
        current_bits = encode_state(current, cfg)
        target_bits = encode_state(target, cfg)
        probs = np.full((dim, cfg.bit_length), 0.5, dtype=float)
        best = current.copy()
        counter["proposal_attempts"] += 1
        best_like = float(likelihood(best[None, :], y, model, counter)[0])
        for generation in range(cfg.generation_count):
            frac = generation / max(1, cfg.generation_count - 1)
            angle = math.pi * (cfg.rotation_angle_max_pi * (1.0 - frac)
                               + cfg.rotation_angle_min_pi * frac)
            bits = np.empty((dim, cfg.bit_length), dtype=np.int8)
            for d in range(dim):
                for b in range(cfg.bit_length):
                    measured = int(rng.random() < probs[d, b])
                    target_bit = int(target_bits[d, b])
                    if measured != target_bit:
                        probs[d, b] = _rotate_probability(probs[d, b], target_bit, angle)
                    bits[d, b] = int(rng.random() < probs[d, b])
            candidate = decode_state(bits, cfg)
            counter["candidate_evals"] += 1
            cand_like = float(likelihood(candidate[None, :], y, model, counter)[0])
            if cand_like >= best_like or rng.random() < min(1.0, cand_like / max(best_like, 1e-300)):
                counter["accepts"] += 1
                best, best_like = candidate, cand_like
        offspring[out_i] = best
    updated = np.vstack([elites, offspring])
    updated_likelihood = likelihood(updated, y, model, counter)
    updated_weights = normalize_logweights(np.log(np.maximum(updated_likelihood, 1e-300)))
    return updated, updated_weights


def run_filter(observations: np.ndarray, truth: np.ndarray, rng: np.random.Generator,
               method: str, model: ModelConfig = ModelConfig(),
               cfg: FilterConfig = FilterConfig()) -> dict[str, object]:
    particles = rng.normal(np.asarray(cfg.initial_mean), np.asarray(cfg.initial_std),
                           size=(cfg.particle_count, 4))
    estimates = np.zeros_like(truth)
    ess_values: list[float] = []
    distinct_values: list[float] = []
    counter = {
        "likelihood_evals": 0,
        "candidate_evals": 0,
        "proposal_attempts": 0,
        "accepts": 0,
    }
    start = time.perf_counter()
    process_std = np.asarray(model.process_std)
    for k in range(1, model.horizon + 1):
        particles = transition(particles) + rng.normal(0.0, process_std, size=particles.shape)
        weights = normalize_logweights(np.log(np.maximum(
            likelihood(particles, observations[k], model, counter), 1e-300
        )))
        estimates[k] = np.sum(weights[:, None] * particles, axis=0)
        ess_values.append(float(1.0 / np.sum(weights ** 2)))
        if method == "bootstrap":
            particles = particles[systematic_resample(weights, rng)]
        elif method == "qgpf":
            particles, weights = qgpf_resample(
                particles, weights, observations[k], rng, model, cfg, counter
            )
        else:
            raise ValueError(f"unknown method: {method}")
        distinct_values.append(float(np.unique(np.round(particles[:, [0, 2]], decimals=8), axis=0).shape[0]))
    errors = estimates[1:] - truth[1:]
    position_error = errors[:, [0, 2]]
    velocity_error = errors[:, [1, 3]]
    return {
        "estimate": estimates,
        "rmse_position": float(np.sqrt(np.mean(position_error ** 2))),
        "rmse_velocity": float(np.sqrt(np.mean(velocity_error ** 2))),
        "rmse_state": float(np.sqrt(np.mean(errors ** 2))),
        "mean_ess": float(np.mean(ess_values)),
        "mean_distinct": float(np.mean(distinct_values)),
        "runtime_s": float(time.perf_counter() - start),
        **counter,
    }

