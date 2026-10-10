"""Source-literal EC-QPF selection kernel for the 180-step runner."""

from __future__ import annotations

import numpy as np


def ellipse_indices(particles: np.ndarray, center: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    particles = np.asarray(particles, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    covariance = np.cov(particles, rowvar=False, ddof=0)
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values <= 0.0):
        raise ValueError("source ellipse requires positive covariance eigenvalues")
    coordinates = (particles - center) @ vectors
    radius = np.sum(coordinates * coordinates / values, axis=1)
    return np.flatnonzero(radius <= scale), covariance, values


def source_literal_logfitness(bits: np.ndarray, distances: np.ndarray, variance: float) -> np.ndarray:
    """Eq. (22) exactly as printed, retaining Euclidean distance rather than squaring it."""
    bits = np.asarray(bits, dtype=np.int8)
    distances = np.asarray(distances, dtype=np.float64)
    counts = bits.sum(axis=1)
    scores = np.full(len(bits), -np.inf, dtype=np.float64)
    valid = counts > 0
    scores[valid] = -np.sum(bits[valid] * distances, axis=1) / (2.0 * variance * counts[valid])
    return scores


def observe(q: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    # Eq. (12): q is retained as the probability of zero.
    return (np.asarray(q) < np.asarray(uniforms)).astype(np.int8)


def update_q(q: np.ndarray, personal: np.ndarray, global_best: np.ndarray, alpha: float, c1: float, c2: float) -> np.ndarray:
    personal_q = alpha * personal + (1.0 - alpha) * (1 - personal)
    global_q = alpha * global_best + (1.0 - alpha) * (1 - global_best)
    return c1 * q + c2 * personal_q + (1.0 - c1 - c2) * global_q


def select_source_literal(
    particles: np.ndarray,
    center: np.ndarray,
    *,
    swarm_size: int,
    iteration_max: int,
    alpha: float,
    c1: float,
    c2: float,
    ellipse_scale: float,
    scalar_variance: float,
    empty_subset_policy: str,
    tie_break_rule: str,
    duplication_allocation: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    eligible, covariance, eigenvalues = ellipse_indices(particles, center, ellipse_scale)
    if not len(eligible):
        raise ValueError("no particle is inside the source error constraint")
    candidates = np.asarray(particles)[eligible]
    distances = np.linalg.norm(candidates - center, axis=1)
    q = rng.random((swarm_size, len(eligible)))
    personal = np.zeros_like(q, dtype=np.int8)
    personal_scores = np.full(swarm_size, -np.inf)
    global_best = np.zeros(len(eligible), dtype=np.int8)
    global_score = -np.inf
    unchanged = 0
    history = []
    for generation in range(iteration_max):
        binary = observe(q, rng.random(q.shape))
        scores = source_literal_logfitness(binary, distances, scalar_variance)
        if empty_subset_policy == "reject":
            pass
        elif empty_subset_policy == "resample":
            empty = ~np.isfinite(scores)
            while np.any(empty):
                binary[empty] = observe(q[empty], rng.random(q[empty].shape))
                scores[empty] = source_literal_logfitness(binary[empty], distances, scalar_variance)
                empty = ~np.isfinite(scores)
        else:
            raise ValueError("empty_subset_policy must be resolved as reject or resample")
        better = scores > personal_scores
        personal[better] = binary[better]
        personal_scores[better] = scores[better]
        if tie_break_rule == "first":
            winner = int(np.argmax(scores))
        elif tie_break_rule == "lexicographic":
            best_score = np.max(scores)
            winner = int(np.flatnonzero(scores == best_score)[0])
        else:
            raise ValueError("tie_break_rule must be resolved")
        if scores[winner] > global_score:
            global_score = float(scores[winner])
            global_best = binary[winner].copy()
            unchanged = 0
        else:
            unchanged += 1
        history.append({"generation": generation + 1, "scores": scores.copy(), "binary": binary.copy(), "global_best": global_best.copy(), "global_score": global_score, "q": q.copy()})
        if unchanged >= max(1, iteration_max // 2):
            break
        q = update_q(q, personal, global_best, alpha, c1, c2)
        if not np.isfinite(q).all() or np.any((q < 0.0) | (q > 1.0)):
            raise FloatingPointError("source probability update left [0,1]")
    selected = eligible[np.flatnonzero(global_best)]
    if not len(selected):
        if empty_subset_policy == "reject":
            raise ValueError("source literal contract produced an empty final subset")
        raise RuntimeError("resample policy failed to produce a nonempty final subset")
    if duplication_allocation == "cyclic":
        chosen = np.resize(selected, len(particles))
    elif duplication_allocation == "balanced":
        chosen = np.tile(selected, int(np.ceil(len(particles) / len(selected))))[: len(particles)]
    else:
        raise ValueError("duplication_allocation must be resolved as cyclic or balanced")
    return {
        "selected_indices": selected,
        "duplicated_indices": chosen,
        "estimate": np.asarray(particles)[chosen].mean(axis=0),
        "ellipse_indices": eligible,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "history": history,
        "fitness_contract": "source_literal_eq22",
        "distance_unit": "U",
        "variance_unit": "U^2 if V is variance",
        "exponent_unit": "U^-1 if V is variance",
    }
