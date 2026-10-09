"""Source operators for an explicitly bounded EC-QPF module audit."""
from dataclasses import dataclass
import time
import numpy as np

@dataclass(frozen=True)
class OperatorConfig:
    particle_count: int = 500
    swarm_size: int = 500
    iteration_max: int = 20
    alpha: float = 0.8
    c1: float = 0.2
    c2: float = 0.2
    ellipse_scale: float = 5.991464547107982

def ellipse_indices(particles, center, scale):
    particles = np.asarray(particles, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[1] != 2 or center.shape != (2,):
        raise ValueError('two-dimensional particles and center required')
    if not np.isfinite(particles).all() or not np.isfinite(center).all() or scale <= 0:
        raise ValueError('finite particles, center, and positive ellipse scale required')
    covariance = np.cov(particles, rowvar=False, ddof=0)
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values <= 0):
        raise ValueError('positive-definite covariance required; no silent regularization')
    # Eq. (21) in principal coordinates; denominators are covariance eigenvalues.
    rotated = (particles - center) @ vectors
    squared_radius = np.sum(rotated**2 / values, axis=1)
    return np.flatnonzero(squared_radius <= scale)

def subset_logfitness(bits, distances, variance):
    bits = np.asarray(bits, dtype=np.int8)
    distances = np.asarray(distances, dtype=np.float64)
    if bits.ndim != 2 or bits.shape[1] != len(distances) or variance <= 0:
        raise ValueError('valid binary subsets, distances, and positive variance required')
    counts = bits.sum(axis=1)
    result = np.full(len(bits), -np.inf, dtype=np.float64)
    valid = counts > 0
    # Eq. (22) contains Euclidean Dis_j, without squaring it.
    result[valid] = -np.sum(bits[valid] * distances, axis=1) / (2.0 * variance * counts[valid])
    return result

def observe_binary(q, uniforms):
    # Literal Eq. (12): q is the probability of bit zero, not bit one.
    return (np.asarray(q) < np.asarray(uniforms)).astype(np.int8)

def update_swarm(q, personal_best, global_best, cfg):
    if not (0 <= cfg.alpha <= 1 and 0 <= cfg.c1 <= 1 and 0 <= cfg.c2 <= 1 and 0 < cfg.c1+cfg.c2 < 1):
        raise ValueError('coefficients violate the source parameter ranges')
    personal_q = cfg.alpha*personal_best + (1.0-cfg.alpha)*(1-personal_best)
    global_q = cfg.alpha*global_best + (1.0-cfg.alpha)*(1-global_best)
    return cfg.c1*q + cfg.c2*personal_q + (1.0-cfg.c1-cfg.c2)*global_q

def duplicate_indices(selected, n):
    selected = np.asarray(selected, dtype=np.int64)
    if not len(selected) or n <= 0:
        raise ValueError('cannot duplicate an empty subset')
    # Source specifies duplication to N, but not its allocation. Audit completion.
    return np.resize(selected, n)

def exact_fitness_subset(distances):
    distances = np.asarray(distances, dtype=np.float64)
    if not len(distances) or not np.isfinite(distances).all():
        raise ValueError('finite nonempty distances required')
    # Any nonempty mean distance >= min(distance); nearest singleton maximizes Eq. (22).
    return np.array([int(np.argmin(distances))], dtype=np.int64)

def _shared_context(particles, center, cfg):
    ellipse = ellipse_indices(particles, center, cfg.ellipse_scale)
    if not len(ellipse):
        raise ValueError('no candidate particles inside shared ellipse')
    candidates = np.asarray(particles)[ellipse]
    distances = np.linalg.norm(candidates-center, axis=1)
    variance = float(np.var(candidates,axis=0,ddof=0).mean())
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError('undefined source-fitness scalar variance')
    return ellipse, distances, variance

def _summarize(particles, ellipse, chosen, distances, variance, start, counts, sampling_ess=None):
    indices = np.asarray(chosen, dtype=np.int64)
    distinct = np.unique(indices)
    source_distances = np.linalg.norm(np.asarray(particles)[distinct]-counts.pop('center'),axis=1)
    return {
        'indices': indices,
        'ellipse_indices': ellipse,
        'estimate': np.asarray(particles)[indices].mean(axis=0),
        'distinct_count': int(len(distinct)),
        'ellipse_count': int(len(ellipse)),
        'logfitness': -float(source_distances.mean())/(2.0*variance),
        'optimal_logfitness': -float(distances.min())/(2.0*variance),
        'fitness_variance': variance,
        'sampling_ess': sampling_ess,
        'runtime_s': time.perf_counter()-start,
        **counts,
    }

def source_select(particles, center, cfg, rng):
    start = time.perf_counter()
    ellipse, distances, variance = _shared_context(particles, center, cfg)
    m = len(ellipse)
    # Persistent probability swarm, observed and updated at every generation.
    q = rng.random((cfg.swarm_size,m))
    personal_best = np.zeros((cfg.swarm_size,m),dtype=np.int8)
    personal_scores = np.full(cfg.swarm_size,-np.inf)
    global_best = np.zeros(m,dtype=np.int8)
    global_score = -np.inf
    unchanged = 0
    empty_subsets = 0
    history = []
    for generation in range(cfg.iteration_max):
        binary = observe_binary(q,rng.random(q.shape))
        scores = subset_logfitness(binary,distances,variance)
        empty_subsets += int(np.sum(~np.isfinite(scores)))
        better = scores > personal_scores
        personal_best[better] = binary[better]
        personal_scores[better] = scores[better]
        winner = int(np.argmax(scores))
        if scores[winner] > global_score:
            global_score = float(scores[winner])
            global_best = binary[winner].copy()
            unchanged = 0
        else:
            unchanged += 1
        if not np.isfinite(global_score):
            raise ValueError('all candidate subsets empty; source fitness unavailable')
        history.append([generation+1,global_score,int(global_best.sum()),float(q.mean())])
        if unchanged >= max(1,cfg.iteration_max//2):
            break
        q = update_swarm(q,personal_best,global_best,cfg)
        if not np.isfinite(q).all() or np.any((q<0)|(q>1)):
            raise FloatingPointError('source probability update left [0,1]')
    selected = ellipse[np.flatnonzero(global_best)]
    indices = duplicate_indices(selected,cfg.particle_count)
    generations = generation+1
    return _summarize(particles,ellipse,indices,distances,variance,start,{
        'center':center,'fitness_evaluations':cfg.swarm_size*generations,
        'candidate_distance_evaluations':m,'binary_dimensions_processed':cfg.swarm_size*m*generations,
        'generations':generations,'empty_subsets':empty_subsets,'likelihood_evaluations':0,
        'generation_history':history,
    })

def shared_classical_select(particles, center, cfg, rng, method):
    start = time.perf_counter()
    ellipse, distances, variance = _shared_context(particles, center, cfg)
    m = len(ellipse)
    sampling_ess = None
    if method == 'ellipse_exact_fitness':
        selected = ellipse[exact_fitness_subset(distances)]
        indices = duplicate_indices(selected,cfg.particle_count)
        score_evals = m
    elif method == 'ellipse_systematic_fitness':
        log_weights = -distances/(2.0*variance)
        weights = np.exp(log_weights-log_weights.max()); weights /= weights.sum()
        cumulative = np.cumsum(weights); cumulative[-1] = 1.0
        positions = (rng.random()+np.arange(cfg.particle_count))/cfg.particle_count
        indices = ellipse[np.searchsorted(cumulative,positions,side='right')]
        sampling_ess = float(1.0/np.sum(weights**2))
        score_evals = m
    else:
        raise ValueError(f'unknown shared-constraint method: {method}')
    return _summarize(particles,ellipse,indices,distances,variance,start,{
        'center':center,'fitness_evaluations':score_evals,'candidate_distance_evaluations':m,
        'binary_dimensions_processed':0,'generations':0,'empty_subsets':0,'likelihood_evaluations':0,
    },sampling_ess)

def full_systematic_select(particles, center, cfg, rng, observation_variance):
    start = time.perf_counter()
    squared = np.sum((np.asarray(particles)-center)**2,axis=1)
    log_weights = -squared/(2.0*observation_variance)
    weights = np.exp(log_weights-log_weights.max()); weights /= weights.sum()
    cumulative = np.cumsum(weights); cumulative[-1] = 1.0
    positions = (rng.random()+np.arange(cfg.particle_count))/cfg.particle_count
    indices = np.searchsorted(cumulative,positions,side='right')
    return {
        'indices':indices,'estimate':np.asarray(particles)[indices].mean(axis=0),
        'sampling_ess':float(1.0/np.sum(weights**2)),
        'distinct_count':int(len(np.unique(indices))),'ellipse_count':None,
        'logfitness':None,'optimal_logfitness':None,'fitness_variance':None,
        'fitness_evaluations':0,'candidate_distance_evaluations':len(particles),
        'binary_dimensions_processed':0,'generations':0,'empty_subsets':0,
        'likelihood_evaluations':len(particles),'runtime_s':time.perf_counter()-start,
    }
