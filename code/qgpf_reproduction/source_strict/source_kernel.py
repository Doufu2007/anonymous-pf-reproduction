"""Source-equation QGPF operators with no undocumented defaults."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


F = np.asarray([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])


def transition(states: np.ndarray) -> np.ndarray:
    return np.asarray(states) @ F.T


def radial_observation(states: np.ndarray) -> np.ndarray:
    return np.sqrt(np.asarray(states)[:, 0] ** 2 + np.asarray(states)[:, 2] ** 2)


def initial_amplitudes(population: int, dimensions: int, bit_length: int) -> np.ndarray:
    value = 1.0 / math.sqrt(2.0)
    return np.full((population, dimensions, bit_length, 2), value, dtype=np.float64)


def measure(amplitudes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    probabilities_one = np.square(amplitudes[..., 1])
    return (rng.random(probabilities_one.shape) < probabilities_one).astype(np.int8)


def rotate_toward(amplitudes: np.ndarray, target: np.ndarray, angle: float, signs: dict[str, float]) -> np.ndarray:
    out = np.asarray(amplitudes, dtype=np.float64).copy()
    for index in np.ndindex(target.shape):
        measured = int(target[index])
        current = int(np.argmax(out[index] ** 2))
        if current == measured:
            continue
        pattern = f"{current}{measured}"
        if pattern not in signs:
            raise ValueError(f"missing source rotation sign for pattern {pattern}")
        theta = float(signs[pattern]) * angle
        c, s = math.cos(theta), math.sin(theta)
        a0, a1 = out[index]
        out[index] = (c * a0 - s * a1, s * a0 + c * a1)
        norm = np.linalg.norm(out[index])
        if not np.isfinite(norm) or norm == 0:
            raise FloatingPointError("invalid source rotation amplitudes")
        out[index] /= norm
    return out


def decode(bits: np.ndarray, state_min: np.ndarray, state_max: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int64)
    powers = 1 << np.arange(bits.shape[-1], dtype=np.int64)
    integers = bits @ powers
    levels = (1 << bits.shape[-1]) - 1
    return np.asarray(state_min) + (integers / levels) * (np.asarray(state_max) - np.asarray(state_min))


def optimise_population(
    particles: np.ndarray,
    fitness: np.ndarray,
    *,
    bit_length: int,
    generations: int,
    state_min: np.ndarray,
    state_max: np.ndarray,
    rotation_max_pi: float,
    rotation_min_pi: float,
    signs: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run the recovered QGA loop; replacement and retention remain caller policy."""
    n, dimensions = particles.shape
    amplitudes = initial_amplitudes(n, dimensions, bit_length)
    best_bits = None
    best_fitness = -np.inf
    traces = []
    for generation in range(generations):
        measured = measure(amplitudes, rng)
        decoded = decode(measured, state_min, state_max)
        current_fitness = np.asarray(fitness(decoded), dtype=np.float64)
        best_index = int(np.argmax(current_fitness))
        if current_fitness[best_index] > best_fitness:
            best_fitness = float(current_fitness[best_index])
            best_bits = measured[best_index].copy()
        fraction = generation / max(1, generations - 1)
        angle = math.pi * (rotation_max_pi * (1.0 - fraction) + rotation_min_pi * fraction)
        amplitudes = rotate_toward(amplitudes, np.broadcast_to(best_bits, measured.shape), angle, signs)
        traces.append({"generation": generation + 1, "measured": measured.copy(), "decoded": decoded.copy(), "fitness": current_fitness.copy(), "best_index": best_index, "rotation_angle": angle, "amplitudes": amplitudes.copy()})
    return {"particles": decode(measure(amplitudes, rng), state_min, state_max), "amplitudes": amplitudes, "traces": traces, "best_fitness": best_fitness}
