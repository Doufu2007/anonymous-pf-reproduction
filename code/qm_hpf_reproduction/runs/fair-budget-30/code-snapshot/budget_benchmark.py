"""Isolated, cached likelihood-budget supplement; old native runs are unchanged.

Only ``resample_move_100`` uses an invariant Metropolis transition. The
QM-HPF-like updates retain the heuristic acceptance/weight convention of the
old independent implementation; they are not posterior-invariant kernels.
"""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from qm_hpf import (FilterConfig, ModelConfig, crossover_candidate, ess,
                    normalize_logweights, particle_logweights,
                    quantum_rotation_candidate, systematic_resample, transition)


METHODS = ("bootstrap_200", "resample_move_100", "qm_hpf_like_100")


def conditional_log_ratio(old, new, ancestor, y, k, model,
                          old_loglikelihood=None, new_loglikelihood=None,
                          process_variance=1.0):
    """Log target ratio for symmetric fixed-ancestor random-walk MH.

    The extended filtering target is proportional to
    w_(k-1)(ancestor) f(x|ancestor) g(y|x). The ancestor weight cancels
    because the move holds the ancestor fixed. No clipping is performed.
    """
    old, new = np.asarray(old), np.asarray(new)
    if old_loglikelihood is None:
        old_loglikelihood = particle_logweights(old, y, k, model.observation_variance)
    if new_loglikelihood is None:
        new_loglikelihood = particle_logweights(new, y, k, model.observation_variance)
    mean = transition(np.asarray(ancestor), k)
    prior_ratio = -((new - mean) ** 2 - (old - mean) ** 2) / (2.0 * process_variance)
    return prior_ratio + np.asarray(new_loglikelihood) - np.asarray(old_loglikelihood)


def resample_pairs(ancestors, particles, loglikelihoods, weights, rng):
    """Apply the same systematic ancestry index to all three arrays."""
    indices = systematic_resample(weights, rng)
    return ancestors[indices].copy(), particles[indices].copy(), loglikelihoods[indices].copy()


def resample_move(particles, ancestors, loglikelihoods, weights, y, k, rng,
                  model, counter, proposal_std=0.1):
    ancestors, moved, cache = resample_pairs(ancestors, particles, loglikelihoods, weights, rng)
    candidates = moved + rng.normal(0.0, proposal_std, size=len(moved))
    candidate_ll = particle_logweights(candidates, y, k, model.observation_variance, counter)
    ratio = conditional_log_ratio(moved, candidates, ancestors, y, k, model, cache, candidate_ll)
    accepted = np.log(rng.random(len(moved))) < np.minimum(0.0, ratio)
    moved[accepted] = candidates[accepted]
    counter["proposal_attempts"] += len(moved)
    counter["candidate_evals"] += len(moved)
    counter["accepts"] += int(accepted.sum())
    return moved, np.full(len(moved), 1.0 / len(moved))


def qm_budget_update(particles, weights, loglikelihoods, y, k, rng, model,
                      cfg, counter, candidate_cap=100):
    """Round-robin low particles, at most G attempts each and cap per step.

    Cached likelihoods eliminate redundant reevaluations of old states.
    Source-inspired asymmetric moves retain the likelihood-only acceptance
    rule, followed by the old weight-times-likelihood-ratio convention. No
    claim of an importance correction or MH invariance follows from this.
    """
    n = len(particles)
    effective = ess(weights)
    order = np.argsort(-weights, kind="stable")
    threshold_index = min(n - 1, max(0, int(np.ceil(effective)) - 1))
    threshold = float(weights[order[threshold_index]])
    high_mask = weights > threshold
    if not np.any(high_mask):
        high_mask[order[0]] = True
    high_indices = np.flatnonzero(high_mask)
    low_indices = np.flatnonzero(~high_mask)
    updated = particles.copy()
    cache = loglikelihoods.copy()
    probability_rotation = 1.0 - effective / n
    attempts = 0
    # G is a true per-low-particle ceiling. A small low set leaves the cap
    # partly unused; neither high particles nor dummy evaluations are added.
    for generation in range(cfg.generation_count):
        for index in low_indices:
            if attempts >= candidate_cap:
                break
            high_index = int(rng.choice(high_indices))
            low = float(updated[index])
            high = float(particles[high_index])
            if rng.random() < probability_rotation:
                candidate = quantum_rotation_candidate(low, high, float(weights[index]),
                    float(weights[high_index]), generation, cfg, rng)
                counter["rotation_proposals"] += 1
            else:
                candidate = crossover_candidate(low, high, effective, n, rng)
                counter["crossover_proposals"] += 1
            candidate_ll = float(particle_logweights(np.array([candidate]), y, k,
                                        model.observation_variance, counter)[0])
            counter["candidate_evals"] += 1
            counter["proposal_attempts"] += 1
            attempts += 1
            # Stable log form of min(1, likelihood(candidate)/likelihood(old)).
            if np.log(rng.random()) < min(0.0, candidate_ll - cache[index]):
                updated[index] = candidate
                cache[index] = candidate_ll
                counter["accepts"] += 1
        if attempts >= candidate_cap:
            break
    with np.errstate(divide="ignore"):
        log_updated_weights = np.log(weights) + cache - loglikelihoods
    updated_weights = normalize_logweights(log_updated_weights)
    indices = systematic_resample(updated_weights, rng)
    return updated[indices], np.full(n, 1.0 / n), attempts


def run_budget_filter(observations, truth, rng, method,
                      model=ModelConfig(), proposal_std=0.1):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    start = time.perf_counter()  # includes particle initialisation
    n = 200 if method == "bootstrap_200" else 100
    cfg = replace(FilterConfig(), particle_count=n)
    particles = rng.normal(model.initial_state, cfg.initial_particle_std, size=n)
    estimates = np.zeros(model.horizon + 1)
    estimates[0] = float(np.mean(particles))
    counter = {"likelihood_evals": 0, "proposal_attempts": 0, "candidate_evals": 0,
               "accepts": 0, "rotation_proposals": 0, "crossover_proposals": 0}
    ess_values, distinct_values, step_counts, step_proposals = [], [], [], []
    weights = np.full(n, 1.0 / n)
    for k in range(1, model.horizon + 1):
        calls_before = counter["likelihood_evals"]
        proposals_before = counter["proposal_attempts"]
        ancestors = particles.copy()
        particles = transition(ancestors, k) + rng.normal(0.0, 1.0, size=n)
        cache = particle_logweights(particles, observations[k], k, model.observation_variance, counter)
        weights = normalize_logweights(cache)
        estimates[k] = float(weights @ particles)  # common pre-move endpoint
        ess_values.append(ess(weights))
        if method == "bootstrap_200":
            particles = particles[systematic_resample(weights, rng)]
            weights = np.full(n, 1.0 / n)
        elif method == "resample_move_100":
            particles, weights = resample_move(particles, ancestors, cache, weights,
                            observations[k], k, rng, model, counter, proposal_std)
        else:
            particles, weights, _ = qm_budget_update(particles, weights, cache,
                            observations[k], k, rng, model, cfg, counter, candidate_cap=100)
        calls = counter["likelihood_evals"] - calls_before
        if calls > 200:
            raise AssertionError(f"step likelihood cap exceeded: {calls}")
        step_counts.append(calls)
        step_proposals.append(counter["proposal_attempts"] - proposals_before)
        distinct_values.append(float(np.unique(np.round(particles, 12)).size))
        if not np.isfinite(particles).all() or not np.isclose(weights.sum(), 1.0):
            raise FloatingPointError("invalid filtering state or weights")
    error = estimates[1:] - truth[1:]
    runtime = float(time.perf_counter() - start)
    return {"estimate": estimates, "final_particles": particles,
            "step_likelihood_evals": np.asarray(step_counts),
            "step_proposal_attempts": np.asarray(step_proposals),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "mean_ess": float(np.mean(ess_values)),
            "mean_distinct": float(np.mean(distinct_values)),
            "runtime_s": runtime, "ess_per_second": float(np.mean(ess_values) / runtime),
            "particle_count": n, "final_weight_sum": float(weights.sum()),
            "budget_shortfall_steps": int(np.sum(np.asarray(step_counts) < 200)),
            "likelihood_cap": 200 * model.horizon, **counter}
