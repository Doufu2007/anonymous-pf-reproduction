"""EC q-zero interpretation: one mapping change, not an original-author claim.

The archived source module is imported read-only for its unchanged ellipse,
variance, distance, exact-optimum and duplication primitives. The new swarm
loop retains its update order and early-stop branch, and adds operation counts.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import time
import numpy as np

HISTORICAL_OPERATOR = Path(__file__).resolve().parents[2] / 'source_aligned_audit' / 'ecqpf_operator.py'
_spec = importlib.util.spec_from_file_location('ec_literal_frozen', HISTORICAL_OPERATOR)
historical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(historical)
OperatorConfig = historical.OperatorConfig
duplicate_indices = historical.duplicate_indices
exact_fitness_subset = historical.exact_fitness_subset
ellipse_indices = historical.ellipse_indices
shared_context = historical._shared_context


def _probability_array(value, name):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError(f'{name} must contain finite probabilities in [0,1]')
    return result


def observe_binary(q, uniforms):
    q = _probability_array(q, 'q'); uniforms = _probability_array(uniforms, 'uniforms')
    if q.shape != uniforms.shape:
        raise ValueError('measurement arrays must have equal shape')
    # Unchanged literal measurement: q=P(bit0), strict inequality observes bit1.
    return (q < uniforms).astype(np.int8)


def _validate_update(q, personal_best, global_best, cfg):
    if not (0 <= cfg.alpha <= 1 and 0 <= cfg.c1 <= 1 and 0 <= cfg.c2 <= 1 and 0 < cfg.c1+cfg.c2 < 1):
        raise ValueError('coefficients violate the frozen source parameter ranges')
    q = _probability_array(q, 'q')
    personal_best = np.asarray(personal_best); global_best = np.asarray(global_best)
    if q.ndim != 2 or personal_best.shape != q.shape or global_best.shape != (q.shape[1],):
        raise ValueError('q/personal best must share a matrix shape and global best its column shape')
    if not np.isin(personal_best, [0, 1]).all() or not np.isin(global_best, [0, 1]).all():
        raise ValueError('best masks must contain only zero and one')
    return q, personal_best, global_best


def update_swarm_consistent(q, personal_best, global_best, cfg):
    q, personal_best, global_best = _validate_update(q, personal_best, global_best, cfg)
    # Declared interpretation: q=P(bit0), so D=0 attracts q toward alpha.
    personal_q = cfg.alpha*(1-personal_best) + (1.0-cfg.alpha)*personal_best
    global_q = cfg.alpha*(1-global_best) + (1.0-cfg.alpha)*global_best
    return cfg.c1*q + cfg.c2*personal_q + (1.0-cfg.c1-cfg.c2)*global_q


def subset_logfitness(bits, distances, variance):
    bits = np.asarray(bits); distances = np.asarray(distances, dtype=np.float64)
    if bits.ndim != 2 or distances.ndim != 1 or bits.shape[1] != len(distances):
        raise ValueError('mask columns must match the distance vector')
    if not np.isin(bits, [0, 1]).all() or not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError('binary masks and finite nonnegative distances required')
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('finite positive subset-independent variance required')
    return historical.subset_logfitness(bits, distances, variance)


def consistent_select(particles, center, cfg, rng):
    start = time.perf_counter()
    particles = np.asarray(particles, dtype=np.float64); center = np.asarray(center, dtype=np.float64)
    if cfg.swarm_size <= 0 or cfg.particle_count <= 0 or cfg.iteration_max <= 0:
        raise ValueError('positive population, output count, and iteration limit required')
    ellipse, distances, variance = shared_context(particles, center, cfg)
    m = len(ellipse)
    q = rng.random((cfg.swarm_size, m))
    personal_best = np.zeros((cfg.swarm_size, m), dtype=np.int8)
    personal_scores = np.full(cfg.swarm_size, -np.inf)
    global_best = np.zeros(m, dtype=np.int8)
    global_score = -np.inf
    unchanged = 0; empty_subsets = 0; q_updates = 0
    personal_replacements = 0; global_replacements = 0
    history = []
    for generation in range(cfg.iteration_max):
        binary = observe_binary(q, rng.random(q.shape))
        scores = subset_logfitness(binary, distances, variance)
        empty_subsets += int(np.sum(~np.isfinite(scores)))
        better = scores > personal_scores
        personal_replacements += int(better.sum())
        personal_best[better] = binary[better]
        personal_scores[better] = scores[better]
        winner = int(np.argmax(scores))
        if scores[winner] > global_score:
            global_score = float(scores[winner]); global_best = binary[winner].copy()
            unchanged = 0; global_replacements += 1
        else:
            unchanged += 1
        if not np.isfinite(global_score):
            raise ValueError('all candidate subsets empty; fail rather than silently fill')
        history.append([generation+1, global_score, int(global_best.sum()), float(q.mean())])
        # Preserve break before update; includes an unused update at max generation.
        if unchanged >= max(1, cfg.iteration_max//2):
            break
        q = update_swarm_consistent(q, personal_best, global_best, cfg)
        q_updates += 1
        if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise FloatingPointError('interpreted probability update left [0,1]')
    selected = ellipse[np.flatnonzero(global_best)]
    indices = duplicate_indices(selected, cfg.particle_count)
    generations = generation+1
    counters = {
        'likelihood_evaluations': 0,
        'fitness_evaluations': cfg.swarm_size*generations,
        'candidate_distance_evaluations': m,
        'summary_distance_evaluations': len(selected),
        'measurement_mask_samples': cfg.swarm_size*generations,
        'measurement_binary_entries': cfg.swarm_size*m*generations,
        'probability_initialization_entries': cfg.swarm_size*m,
        'probability_update_entries': cfg.swarm_size*m*q_updates,
        'personal_best_score_comparisons': cfg.swarm_size*generations,
        'personal_best_replacements': personal_replacements,
        'global_argmax_input_scores': cfg.swarm_size*generations,
        'global_best_score_comparisons': generations,
        'global_best_replacements': global_replacements,
        'duplication_output_indices': cfg.particle_count,
        'rotation_events': 0, 'crossover_attempts': 0,
        'oracle_calls': 0, 'resampling_output_indices': 0,
    }
    result = historical._summarize(particles, ellipse, indices, distances, variance, start, {
        'center': center, 'fitness_evaluations': cfg.swarm_size*generations,
        'candidate_distance_evaluations': m,
        'binary_dimensions_processed': cfg.swarm_size*m*generations,
        'generations': generations, 'empty_subsets': empty_subsets,
        'likelihood_evaluations': 0, 'generation_history': history,
        'probability_update_steps': q_updates, 'subset_size': int(global_best.sum()),
        'operation_counts': counters,
    })
    if result['distinct_count'] != result['subset_size'] or result['logfitness'] > result['optimal_logfitness']+1e-12:
        raise AssertionError('duplication or exact-fitness bound violated')
    return result
