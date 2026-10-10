"""Declared 3D MR.CLAM audit contract, not a source-QGPF reproduction.

Inputs contain real odometry and range/bearing observations, a known static
landmark map, and map-derived bounds. No evaluation truth reaches this module.
The Q mechanisms are transferred heuristics without an importance-proposal
correction. Pre-move estimates assimilate the current observation; final Q
likelihood weights are only post-move diagnostics and, as in the historical
audit contract, are discarded before the next prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np


METHODS = ('B-stable', 'RM-stable', 'Q-stable', 'Q-stable-retain')
COUNTER_NAMES = ('likelihood_evals', 'scalar_observation_evals', 'candidate_evals',
                 'proposal_candidate_attempts', 'proposal_chromosome_events',
                 'rotation_calls', 'crossover_operations', 'measurement_bit_observations',
                 'probability_entry_updates', 'fitness_evals', 'accepts', 'rejects',
                 'resampling_calls', 'motion_state_segment_evals')


@dataclass(frozen=True)
class Config:
    budget: int = 1200
    quantum_particles: int = 200
    process_std_per_sqrt_s: tuple = (.05, .05, .03)
    range_std_m: float = .2
    bearing_std_rad: float = .1
    state_min: tuple = (-20., -20., -math.pi)
    state_max: tuple = (20., 20., math.pi)
    bit_length: int = 10
    generation_count: int = 4
    elite_fraction: float = .2
    rotation_angle_max_pi: float = .08
    rotation_angle_min_pi: float = .01


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2. * np.pi) - np.pi


def normalize_logweights(logw):
    a = np.asarray(logw, dtype=np.float64)
    if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all():
        raise FloatingPointError('finite nonempty scalar log weights required')
    weights = np.exp(a - np.max(a))
    weights /= weights.sum()
    if not np.isfinite(weights).all():
        raise FloatingPointError('nonfinite normalized weights')
    return weights


def advance_unicycle(states, segments):
    """Exact piecewise-constant velocity arc integration, row=(dt,v,w)."""
    p = np.asarray(states, dtype=np.float64).copy()
    s = np.asarray(segments, dtype=np.float64).reshape(-1, 3)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('finite N by 3 states required')
    if not np.isfinite(s).all() or np.any(s[:, 0] < 0.):
        raise ValueError('finite nonnegative-duration odometry required')
    for dt, v, w in s:
        # sinc avoids the zero-angular-velocity division and is algebraically
        # identical to the circular arc under constant controls.
        turn = float(w * dt)
        distance = float(v * dt * np.sinc(turn / (2. * np.pi)))
        mid_heading = p[:, 2] + turn / 2.
        p[:, 0] += distance * np.cos(mid_heading)
        p[:, 1] += distance * np.sin(mid_heading)
        p[:, 2] = wrap_angle(p[:, 2] + turn)
    return p


def log_likelihood(states, landmark_xy, observations, cfg, counter=None):
    """One state-event likelihood; independent actual range/bearing rows.

    likelihood_evals counts complete state/batch evaluations. The separate
    scalar_observation_evals counts individual range-bearing-pair evaluations;
    multiplying by two would count univariate Gaussian residual terms.
    """
    p = np.asarray(states, dtype=np.float64)
    xy = np.asarray(landmark_xy, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(observations, dtype=np.float64).reshape(-1, 2)
    if (p.ndim != 2 or p.shape[1] != 3 or xy.shape != z.shape or len(z) == 0
            or not all(np.isfinite(a).all() for a in (p, xy, z))
            or cfg.range_std_m <= 0. or cfg.bearing_std_rad <= 0.):
        raise ValueError('finite states, matched nonempty observation/map rows and positive noise required')
    if counter is not None:
        counter['likelihood_evals'] += len(p)
        counter['scalar_observation_evals'] += len(p) * len(z)
    delta = xy[None, :, :] - p[:, None, :2]
    ranges = np.sqrt(np.sum(delta ** 2, axis=2))
    bearings = wrap_angle(np.arctan2(delta[:, :, 1], delta[:, :, 0]) - p[:, 2, None])
    range_residual = z[None, :, 0] - ranges
    bearing_residual = wrap_angle(z[None, :, 1] - bearings)
    # Wrapped shortest-residual Gaussian is a declared common observation
    # score. It is not claimed to be the exact wrapped-normal likelihood.
    with np.errstate(over='raise', invalid='raise'):
        ll = -.5 * np.sum((range_residual / cfg.range_std_m) ** 2
                         + (bearing_residual / cfg.bearing_std_rad) ** 2, axis=1)
    ll -= len(z) * math.log(2. * math.pi * cfg.range_std_m * cfg.bearing_std_rad)
    if not np.isfinite(ll).all():
        raise FloatingPointError('nonfinite log likelihood')
    return ll


def weighted_pose_mean(particles, weights):
    p, w = np.asarray(particles), np.asarray(weights)
    xy = np.sum(w[:, None] * p[:, :2], axis=0)
    heading = math.atan2(float(np.sum(w * np.sin(p[:, 2]))),
                         float(np.sum(w * np.cos(p[:, 2]))))
    return np.array([xy[0], xy[1], heading], dtype=np.float64)


def systematic_resample(weights, rng):
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.
    return np.searchsorted(cumulative, positions, side='right')


def accept_independence_prior_move(log_old, log_new, uniform):
    """MH acceptance when the proposal is the same transition prior.

    The transition-prior factors cancel from target/proposal, leaving the
    observation likelihood ratio. This applies only to RM-stable.
    """
    old, new, u = float(log_old), float(log_new), float(uniform)
    if not all(math.isfinite(x) for x in (old, new)) or not 0. <= u < 1.:
        raise ValueError('finite log likelihoods and uniform in [0,1) required')
    return new >= old or math.log(u) < new - old


def encode_states(states, cfg):
    lo, hi = np.asarray(cfg.state_min), np.asarray(cfg.state_max)
    levels = (1 << cfg.bit_length) - 1
    ints = np.rint((np.clip(states, lo, hi) - lo) / (hi - lo) * levels).astype(np.int64)
    return ((ints[:, :, None] >> np.arange(cfg.bit_length)) & 1).astype(np.int8)


def decode_states(bits, cfg):
    lo, hi = np.asarray(cfg.state_min), np.asarray(cfg.state_max)
    ints = np.sum(bits * (1 << np.arange(cfg.bit_length)), axis=2)
    states = lo + ints / ((1 << cfg.bit_length) - 1) * (hi - lo)
    states[:, 2] = wrap_angle(states[:, 2])
    return states


def qgpf_update(particles, weights, landmark_xy, observations, cfg, counter, rng, retain):
    """Vectorized transfer of classical Bernoulli-amplitude rotations.

    Vectorization changes the RNG schedule relative to the historical 4D
    implementation. Both new Q variants use the same fixed-size RNG draws
    in each generation. Encoding is 3D, map-bounded, with wrapped heading.
    """
    n, dim = particles.shape
    elite_n = int(round(cfg.elite_fraction * n))
    if not 0 < elite_n < n:
        raise ValueError('at least one elite and nonelite required')
    order = np.argsort(-weights, kind='stable')
    elites = particles[order[:elite_n]].copy()
    elite_weights = weights[order[:elite_n]]
    parent_indices = rng.choice(elite_n, size=n - elite_n, p=elite_weights / elite_weights.sum())
    targets = encode_states(elites[parent_indices], cfg)
    probabilities = np.full((n - elite_n, dim, cfg.bit_length), .5)
    best = particles[order[elite_n:]].copy()
    counter['proposal_chromosome_events'] += len(best)
    best_log = log_likelihood(best, landmark_xy, observations, cfg, counter)
    for generation in range(cfg.generation_count):
        frac = generation / max(1, cfg.generation_count - 1)
        angle = math.pi * (cfg.rotation_angle_max_pi * (1. - frac)
                          + cfg.rotation_angle_min_pi * frac)
        measured = rng.random(probabilities.shape) < probabilities
        rotate = measured != targets
        a0, a1 = np.sqrt(1. - probabilities), np.sqrt(probabilities)
        c, s = math.cos(angle), math.sin(angle)
        next_a1 = np.where(targets == 1, s * a0 + c * a1, -s * a0 + c * a1)
        probabilities = np.where(rotate, np.clip(next_a1 ** 2, 0., 1.), probabilities)
        bits = (rng.random(probabilities.shape) < probabilities).astype(np.int8)
        candidate = decode_states(bits, cfg)
        candidate_log = log_likelihood(candidate, landmark_xy, observations, cfg, counter)
        uniforms = rng.random(len(candidate))  # consumed by both variants
        with np.errstate(divide='ignore'):
            log_uniforms = np.log(uniforms)  # U=0 means log U=-inf; no probability floor.
        accepted = (candidate_log > best_log if retain else
                    log_uniforms < np.minimum(0., candidate_log - best_log))
        best[accepted] = candidate[accepted]
        best_log[accepted] = candidate_log[accepted]
        counter['rotation_calls'] += int(rotate.sum())
        counter['probability_entry_updates'] += int(rotate.sum())
        counter['measurement_bit_observations'] += int(2 * probabilities.size)
        for name in ('candidate_evals', 'fitness_evals', 'proposal_candidate_attempts'):
            counter[name] += len(candidate)
        counter['accepts'] += int(accepted.sum())
        counter['rejects'] += int(len(candidate) - accepted.sum())
    updated = np.vstack([elites, best])
    post_weights = normalize_logweights(log_likelihood(updated, landmark_xy, observations, cfg, counter))
    return updated, post_weights


def run_filter(odometry_segments, landmark_xy, observations, initial_center,
               method, seed, cfg=Config()):
    """Run map-only global initialization and real-input filtering.

    initial_center exists for loader compatibility but is explicitly unused;
    global uniform initialization never reads robot ground truth.
    Resample-move conditions on the retained predictive ancestor mean. The
    proposal redraws its transition prior, so its density and the target's
    transition-prior factor cancel; the exact MH ratio is the observation
    likelihood ratio under this declared model.
    """
    if method not in METHODS:
        raise ValueError('unknown method')
    if len(odometry_segments) != len(observations) or len(landmark_xy) != len(observations):
        raise ValueError('matched event streams required')
    n = cfg.budget if method == 'B-stable' else cfg.budget // 2 if method == 'RM-stable' else cfg.quantum_particles
    if method == 'RM-stable' and cfg.budget % 2:
        raise ValueError('RM budget must be divisible by two')
    elite_n = int(round(cfg.elite_fraction * n))
    qcost = 2 * n + (n - elite_n) * (1 + cfg.generation_count)
    if method.startswith('Q-') and qcost != cfg.budget:
        raise ValueError('Q event budget is not exact')
    lo, hi = np.asarray(cfg.state_min), np.asarray(cfg.state_max)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(lo >= hi):
        raise ValueError('3D map bounds required')
    rng = np.random.default_rng(int(seed))
    start = time.perf_counter()
    particles = rng.uniform(lo, hi, size=(n, 3))
    particles[:, 2] = wrap_angle(particles[:, 2])
    initial_particles = particles.copy()
    init_s = time.perf_counter() - start
    count = {name: 0 for name in COUNTER_NAMES}
    pre, post, step_counts, ess, distinct, elapsed = [], [], [], [], [], []
    normalized = True
    for segments, xy, z in zip(odometry_segments, landmark_xy, observations):
        tick = time.perf_counter()
        before = dict(count)
        segments = np.asarray(segments, dtype=float).reshape(-1, 3)
        predicted_means = advance_unicycle(particles, segments)
        count['motion_state_segment_evals'] += n * len(segments)
        sigma = np.asarray(cfg.process_std_per_sqrt_s) * math.sqrt(float(segments[:, 0].sum()))
        particles = predicted_means + rng.normal(0., sigma, size=particles.shape)
        particles[:, 2] = wrap_angle(particles[:, 2])
        base_log = log_likelihood(particles, xy, z, cfg, count)
        weights = normalize_logweights(base_log)
        pre.append(weighted_pose_mean(particles, weights))
        ess.append(float(1. / np.sum(weights ** 2)))
        normalized &= bool(np.isclose(weights.sum(), 1., rtol=0., atol=1e-12))
        if method in ('B-stable', 'RM-stable'):
            indices = systematic_resample(weights, rng)
            count['resampling_calls'] += 1
            particles = particles[indices].copy()
            post_weights = np.full(n, 1. / n)
            if method == 'RM-stable':
                # The resampled (ancestor,current) pair targets the augmented
                # posterior. An independence-prior update conditional on the
                # ancestor is invariant to that conditional posterior.
                candidates = predicted_means[indices] + rng.normal(0., sigma, size=particles.shape)
                candidates[:, 2] = wrap_angle(candidates[:, 2])
                candidate_log = log_likelihood(candidates, xy, z, cfg, count)
                uniform = rng.random(n)
                accepted = np.array([accept_independence_prior_move(old, new, u)
                                     for old, new, u in zip(base_log[indices], candidate_log, uniform)])
                particles[accepted] = candidates[accepted]
                for name in ('candidate_evals', 'proposal_candidate_attempts'):
                    count[name] += n
                count['accepts'] += int(accepted.sum())
                count['rejects'] += int(n - accepted.sum())
        else:
            particles, post_weights = qgpf_update(particles, weights, xy, z, cfg, count,
                                                 rng, method == 'Q-stable-retain')
        post.append(weighted_pose_mean(particles, post_weights))
        distinct.append(int(np.unique(np.round(particles, 8), axis=0).shape[0]))
        normalized &= bool(np.isclose(post_weights.sum(), 1., rtol=0., atol=1e-12))
        step_counts.append([count[name] - before[name] for name in COUNTER_NAMES])
        elapsed.append(time.perf_counter() - tick)
        if not np.isfinite(particles).all():
            raise FloatingPointError('nonfinite particles')
    pre, post = np.asarray(pre), np.asarray(post)
    step_counts = np.asarray(step_counts, dtype=np.int64)
    guardrails = {'finite_estimates': bool(np.isfinite(pre).all() and np.isfinite(post).all()),
                  'all_weights_normalized': normalized,
                  'counter_sum_matches': bool(np.array_equal(step_counts.sum(axis=0), [count[x] for x in COUNTER_NAMES])),
                  'exact_event_likelihood_budget': bool(np.all(step_counts[:, 0] == cfg.budget)),
                  'truth_unavailable_to_filter': True,
                  'map_only_initialization': True}
    if not all(guardrails.values()):
        raise AssertionError('filter guardrail failure')
    return {'estimate_pre': pre, 'estimate_post': post,
            'initial_particles': initial_particles, 'final_particles': particles.copy(),
            'final_weights': post_weights.copy(), 'step_counters': step_counts,
            'step_ess': np.asarray(ess), 'step_distinct': np.asarray(distinct, dtype=np.int64),
            'step_elapsed_s': np.asarray(elapsed), 'runtime_s': time.perf_counter() - start,
            'initialization_runtime_s': init_s, 'counters': count, 'guardrails': guardrails,
            'rng_final_state': rng.bit_generator.state, 'particle_count': n}
