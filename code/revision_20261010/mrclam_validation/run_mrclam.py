"""Immutable real-input MR.CLAM runs under a frozen, map-only audit contract."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from algorithm import Config, COUNTER_NAMES, METHODS, run_filter, wrap_angle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_CACHE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def input_hashes(sequence):
    # Ragged event arrays are hashed as the ordered component hash list,
    # never pickled. The evaluator truth receives its own disjoint hash.
    result = {}
    for key in ('event_times', 'truth', 'known_map'):
        if key in sequence:
            result[key] = array_sha(sequence[key])
    for key in ('odometry_segments', 'observations', 'landmark_xy'):
        component_hashes = [array_sha(x) for x in sequence[key]]
        result[key] = hashlib.sha256(json.dumps(component_hashes).encode()).hexdigest()
    return result


def git_provenance():
    try:
        def git(*args):
            return subprocess.run(['git', *args], cwd=REPO, text=True, check=True,
                                  capture_output=True).stdout.strip()
        state = git('status', '--porcelain').splitlines()
        return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(state), 'status': state}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {'available': False, 'error': str(exc)}


def load_cached(raw_dir, dataset_id, robot_id, duration):
    key = (str(raw_dir), dataset_id, robot_id, duration)
    if key not in _CACHE:
        from data import load_sequence
        _CACHE[key] = load_sequence(raw_dir, dataset_id, robot_id, duration_s=duration)
    return _CACHE[key]


def sequence_bounds(sequence):
    if 'state_min' in sequence and 'state_max' in sequence:
        return tuple(np.asarray(sequence['state_min']).tolist()), tuple(np.asarray(sequence['state_max']).tolist())
    xy = np.asarray(sequence['known_map'], dtype=float)
    return (float(xy[:, 0].min() - 1.), float(xy[:, 1].min() - 1.), -float(np.pi)), (
        float(xy[:, 0].max() + 1.), float(xy[:, 1].max() + 1.), float(np.pi))


def run_one(raw_dir, dataset_id, robot_id, seed, method, duration, out_dir,
            protocol_path, execution_workers=1):
    run_dir = Path(out_dir) / f'dataset-{dataset_id:02d}' / f'robot-{robot_id}' / method / f'seed-{seed:02d}'
    if run_dir.exists():
        raise FileExistsError(f'immutable run directory exists: {run_dir}')
    run_dir.mkdir(parents=True)
    started = now()
    metrics, failure = {}, None
    protocol = json.loads(Path(protocol_path).read_text(encoding='utf-8'))
    protocol_hash = sha(protocol_path)
    status = 'failed'
    try:
        sequence = load_cached(raw_dir, dataset_id, robot_id, duration)
        lo, hi = sequence_bounds(sequence)
        config_dict = dict(protocol.get('algorithm_config', {}))
        # Map bounds replace only the declared map-derived fields. This is
        # deterministic external input preprocessing, not GT-based tuning.
        config_dict['state_min'], config_dict['state_max'] = lo, hi
        cfg = Config(**config_dict)
        config_record = {'dataset_id': dataset_id, 'robot_id': robot_id, 'seed': seed,
                         'filter_seed': 710000 + seed, 'method': method, 'duration_s': duration,
                         'filter': asdict(cfg), 'protocol_sha256': protocol_hash,
                         'initialization': 'uniform known-map bounding box plus 1m and heading [-pi,pi)',
                         'primary_endpoint': 'radial position RMSE of post-assimilation/pre-move estimates at selected events',
                         'secondary_endpoint': 'wrapped heading RMSE; post-move radial RMSE; ESS; distinct poses; runtime',
                         'code_sha256': {name: sha(HERE / name) for name in ('algorithm.py', 'run_mrclam.py', 'data.py')},
                         'execution_workers': execution_workers,
                         'parallel_wallclock_not_strictly_comparable': execution_workers > 1,
                         'loader_provenance': sequence.get('metadata', {})}
        dump(run_dir / 'config.json', config_record)
        dump(run_dir / 'input_hashes.json', input_hashes(sequence))
        dump(run_dir / 'environment.json', {'python': sys.version, 'numpy': np.__version__,
             'platform': platform.platform(), 'cpu': platform.processor(), 'device': 'CPU',
             'dtype': 'float64', 'threads': {k: os.environ.get(k) for k in
             ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
             'git': git_provenance(), 'timer_scope': 'initialization, filter execution, in-memory diagnostics; excludes file parsing/evaluation/output IO'})
        # This call is the leakage boundary. No whole sequence dictionary,
        # truth array or true initial pose is supplied to the algorithm.
        from data import filter_inputs
        inputs = filter_inputs(sequence)
        result = run_filter(inputs['odometry_segments'], inputs['landmark_xy'],
                            inputs['observations'], None, method, 710000 + seed, cfg)
        truth = np.asarray(sequence['truth'], dtype=float)
        pre, post = result['estimate_pre'], result['estimate_post']
        valid = np.asarray(sequence.get('truth_valid', np.isfinite(truth).all(axis=1)), dtype=bool)
        if truth.shape != pre.shape or valid.shape != (len(pre),) or not valid.all() or not np.isfinite(truth).all():
            raise ValueError('MR.CLAM evaluation requires complete ground truth at every retained event; no event may be silently dropped')
        error = pre[valid, :2] - truth[valid, :2]
        radius_error = np.sqrt(np.sum(error ** 2, axis=1))
        heading_error = wrap_angle(pre[valid, 2] - truth[valid, 2])
        metrics = {'dataset_id': dataset_id, 'robot_id': robot_id, 'seed': seed, 'method': method,
                   'event_count': len(pre), 'particle_count': result['particle_count'],
                   'scored_event_count': int(valid.sum()), 'truth_coverage_fraction': float(valid.mean()),
                   'rmse_position_radial_m': float(np.sqrt(np.mean(radius_error ** 2))),
                   'mae_position_radial_m': float(np.mean(radius_error)),
                   'rmse_position_per_coordinate_m': float(np.sqrt(np.mean(error ** 2))),
                   'rmse_heading_rad': float(np.sqrt(np.mean(heading_error ** 2))),
                   'post_diagnostic_rmse_position_radial_m': float(np.sqrt(np.mean(np.sum((post[valid, :2] - truth[valid, :2]) ** 2, axis=1)))),
                   'mean_ess': float(np.mean(result['step_ess'])),
                   'mean_ess_fraction': float(np.mean(result['step_ess']) / result['particle_count']),
                   'mean_distinct': float(np.mean(result['step_distinct'])),
                   'runtime_s': result['runtime_s'], 'initialization_runtime_s': result['initialization_runtime_s'],
                   **{k: int(v) for k, v in result['counters'].items()}}
        for key in ('estimate_pre', 'estimate_post', 'initial_particles', 'final_particles', 'final_weights',
                    'step_counters', 'step_ess', 'step_distinct', 'step_elapsed_s'):
            np.save(run_dir / f'{key}.npy', result[key], allow_pickle=False)
        np.save(run_dir / 'truth.npy', truth, allow_pickle=False)
        np.save(run_dir / 'truth_valid.npy', valid, allow_pickle=False)
        np.save(run_dir / 'known_map.npy', sequence['known_map'], allow_pickle=False)
        np.save(run_dir / 'event_times.npy', sequence['event_times'], allow_pickle=False)
        # All ragged real observations are retained in a portable compressed
        # numeric bundle with offsets. No ground truth is used as observations.
        for key in ('observations', 'landmark_xy', 'odometry_segments'):
            values = sequence[key]
            offsets = np.concatenate([[0], np.cumsum([len(x) for x in values])])
            np.savez_compressed(run_dir / f'{key}.npz', values=np.vstack(values), offsets=offsets)
        dump(run_dir / 'raw_metrics.json', metrics)
        dump(run_dir / 'guardrails.json', result['guardrails'])
        dump(run_dir / 'rng_final_state.json', result['rng_final_state'])
        with (run_dir / 'predictions.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['time_s', *[f'{label}_{axis}' for label in ('truth', 'pre_move', 'post_move') for axis in ('x', 'y', 'theta')]])
            writer.writerows([float(t), *actual, *p, *q] for t, actual, p, q in zip(sequence['event_times'], truth, pre, post))
        status = 'completed'
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
                   'failure_preserved': True, 'seed_not_excluded': True}
        dump(run_dir / 'failure.json', failure)
    (run_dir / 'stdout.log').write_text(json.dumps({'status': status, 'metrics': metrics}, ensure_ascii=False) + '\n', encoding='utf-8')
    (run_dir / 'stderr.log').write_text(json.dumps(failure, ensure_ascii=False) + '\n' if failure else '', encoding='utf-8')
    files = {p.name: sha(p) for p in run_dir.iterdir() if p.is_file() and p.name != 'manifest.json'}
    manifest = {'schema': 'mrclam.run.v1', 'dataset_id': dataset_id, 'robot_id': robot_id, 'seed': seed,
                'method': method, 'status': status, 'started_at': started, 'ended_at': now(),
                'protocol_sha256': protocol_hash, 'files_sha256': files, 'failure': failure}
    dump(run_dir / 'manifest.json', manifest)
    return {'run_dir': str(run_dir), 'status': status, 'metrics': metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', required=True)
    parser.add_argument('--datasets', type=int, nargs='+', default=[8, 9])
    parser.add_argument('--robot', type=int, default=1)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(30)))
    parser.add_argument('--methods', choices=METHODS, nargs='+', default=list(METHODS))
    parser.add_argument('--duration-s', type=float, default=1200.)
    parser.add_argument('--out', required=True)
    parser.add_argument('--protocol', default=str(HERE / 'protocol.json'))
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'immutable batch directory exists: {out}')
    out.mkdir(parents=True)
    tasks = [(args.raw_dir, d, args.robot, s, m, args.duration_s, str(out), args.protocol, args.workers)
             for d in args.datasets for s in args.seeds for m in args.methods]
    results = []
    if args.workers == 1:
        for task in tasks:
            result = run_one(*task)
            results.append(result)
            print(json.dumps({k: result[k] for k in ('run_dir', 'status')}, ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, *task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({k: result[k] for k in ('run_dir', 'status')}, ensure_ascii=False), flush=True)
    results.sort(key=lambda x: x['run_dir'])
    completed = [x for x in results if x['status'] == 'completed']
    failures = [x for x in results if x['status'] != 'completed']
    rows = [x['metrics'] for x in completed]
    if rows:
        with (out / 'raw_metrics.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    dump(out / 'batch_manifest.json', {'planned_runs': len(tasks), 'completed_runs': len(completed),
         'failed_runs': len(failures), 'results': results, 'args': vars(args),
         'no_failed_seed_exclusion': True, 'mc_seeds_are_not_independent_real_trajectories': True,
         'protocol_sha256': sha(args.protocol)})
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
