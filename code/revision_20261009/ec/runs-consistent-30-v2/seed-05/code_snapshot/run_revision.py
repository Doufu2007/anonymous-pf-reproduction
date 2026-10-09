"""Run one EC interpretation on archived inputs; retain all 30 datasets.

Only ec_qzero_consistent is newly executed. Literal/exact/systematic arrays and
their scientific metrics are verified and reused from the frozen historical
batch. Historical timing is retained for provenance, never a current speed rank.
"""
from __future__ import annotations
import argparse
import csv
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from ecqpf_interpreted import OperatorConfig, consistent_select, shared_context, HISTORICAL_OPERATOR

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parents[3]
HISTORICAL = BASE.parents[1] / 'source_aligned_audit'
INPUT_ROOT = HISTORICAL / 'runs-main-30'
PROTOCOL = WORKSPACE / 'revision-2026-10-09' / 'frozen-protocol.json'
FREEZE = WORKSPACE / 'revision-2026-10-09' / 'historical-freeze.json'
NEW_METHOD = 'ec_qzero_consistent'
OLD_METHODS = ['source_ecqpf_operator', 'ellipse_exact_fitness', 'ellipse_systematic_fitness']
METHODS = [NEW_METHOD] + OLD_METHODS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def json_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def verify_frozen(path, freeze):
    key = Path(path).relative_to(WORKSPACE).as_posix()
    found = digest(path)
    if key not in freeze['inputs'] or found != freeze['inputs'][key]:
        raise AssertionError(f'historical hash mismatch: {key}')
    return {'path': key, 'sha256': found}


def validate_protocol():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8-sig'))
    freeze = json.loads(FREEZE.read_text(encoding='utf-8-sig'))
    if digest(FREEZE) != protocol['historical_freeze']['sha256']:
        raise AssertionError('historical-freeze record differs from frozen protocol')
    ec = protocol['EC']; cfg = OperatorConfig()
    expected = {'N': cfg.particle_count, 'population': cfg.swarm_size, 'alpha': cfg.alpha,
                'c1': cfg.c1, 'c2': cfg.c2, 'iteration_limit': cfg.iteration_max,
                'unchanged_best_stop': max(1, cfg.iteration_max//2)}
    for key, val in expected.items():
        if ec[key] != val:
            raise AssertionError(f'EC protocol/config mismatch: {key}')
    if ec['seeds'] != list(range(30)) or ec['updates_per_dataset'] != 20:
        raise AssertionError('only the frozen 30 datasets x20 updates are supported')
    if ec['statistics']['bootstrap'] != 20000 or ec['statistics']['sign_flip'] != 50000:
        raise AssertionError('analysis replication counts differ from frozen protocol')
    verify_frozen(HISTORICAL_OPERATOR, freeze)
    return protocol, freeze, cfg


def verify_historical_seed(seed, freeze, cfg):
    source = INPUT_ROOT / f'seed-{seed:02d}'
    names = ['truth.npy', 'centers.npy', 'particles.npy', 'config.json', 'raw_metrics.jsonl',
             'seed_summary.json', 'costs.json', 'manifest.json', 'source_generation_history.json']
    for method in OLD_METHODS:
        names += [f'{method}_estimates.npy', f'{method}_indices.npy']
    hashes = [verify_frozen(source/name, freeze) for name in names]
    oldconfig = json.loads((source/'config.json').read_text(encoding='utf-8-sig'))
    if oldconfig['operator'] != asdict(cfg) or oldconfig['plan']['updates_per_seed'] != 20:
        raise AssertionError('historical operator contract mismatch')
    if oldconfig['plan']['methods'][0] != 'source_ecqpf_operator':
        raise AssertionError('historical source stream is not stream3')
    truth = np.load(source/'truth.npy', allow_pickle=False)
    centers = np.load(source/'centers.npy', allow_pickle=False)
    clouds = np.load(source/'particles.npy', allow_pickle=False)
    if truth.shape != (20, 2) or centers.shape != (20, 2) or clouds.shape != (20, 500, 2):
        raise AssertionError('historical input shape mismatch')
    if not all(np.isfinite(a).all() for a in [truth, centers, clouds]):
        raise AssertionError('nonfinite historical inputs')
    oldrows = [r for r in json_rows(source/'raw_metrics.jsonl') if r['method'] in OLD_METHODS]
    if len(oldrows) != 60 or len({(r['method'], r['update']) for r in oldrows}) != 60:
        raise AssertionError('incomplete or duplicated historical comparison rows')
    est = {}; idx = {}
    summary = {r['method']: r for r in json.loads((source/'seed_summary.json').read_text(encoding='utf-8'))}
    for method in OLD_METHODS:
        est[method] = np.load(source/f'{method}_estimates.npy', allow_pickle=False)
        idx[method] = np.load(source/f'{method}_indices.npy', allow_pickle=False)
        if est[method].shape != (20, 2) or idx[method].shape != (20, 500):
            raise AssertionError('historical output shape mismatch')
        if idx[method].dtype.kind not in 'iu' or np.any((idx[method] < 0) | (idx[method] >= 500)):
            raise AssertionError('invalid historical indices')
        recomputed = np.stack([clouds[j][idx[method][j]].mean(axis=0) for j in range(20)])
        np.testing.assert_allclose(recomputed, est[method], rtol=0, atol=1e-14)
        rmse = float(np.sqrt(np.mean(np.sum((est[method]-truth)**2, axis=1))))
        np.testing.assert_allclose(rmse, summary[method]['rmse'], rtol=0, atol=1e-14)
        for row in [r for r in oldrows if r['method'] == method]:
            j = row['update']
            np.testing.assert_allclose(row['squared_state_error'], np.sum((est[method][j]-truth[j])**2), rtol=0, atol=1e-12)
    return source, hashes, truth, centers, clouds, oldrows, est, idx


def historical_operations(row):
    """Known nonapplicable operations are zero; unrecorded applicable ones null."""
    method = row['method']; swarm = method == 'source_ecqpf_operator'
    systematic = method == 'ellipse_systematic_fitness'
    return {
        'likelihood_evaluations': row['likelihood_evaluations'],
        'fitness_evaluations': row['fitness_evaluations'],
        'candidate_distance_evaluations': row['candidate_distance_evaluations'],
        'summary_distance_evaluations': row['distinct_count'],
        'measurement_mask_samples': row['fitness_evaluations'] if swarm else 0,
        'measurement_binary_entries': row['binary_dimensions_processed'],
        'probability_initialization_entries': 500*row['ellipse_count'] if swarm else 0,
        'probability_update_entries': None if swarm else 0,
        'personal_best_score_comparisons': row['fitness_evaluations'] if swarm else 0,
        'personal_best_replacements': None if swarm else 0,
        'global_argmax_input_scores': row['fitness_evaluations'] if swarm else 0,
        'global_best_score_comparisons': row['generations'] if swarm else 0,
        'global_best_replacements': None if swarm else 0,
        'duplication_output_indices': 0 if systematic else 500,
        'rotation_events': 0, 'crossover_attempts': 0, 'oracle_calls': 0,
        'resampling_output_indices': 500 if systematic else 0,
    }


def summarize(seed, rows, costs):
    summaries = []
    for method in METHODS:
        rr = [r for r in rows if r['method'] == method]
        cc = [r for r in costs if r['method'] == method]
        operations = {}
        for key in rr[0]['operation_counts']:
            values = [r['operation_counts'][key] for r in rr]
            operations[key] = None if any(v is None for v in values) else sum(values)
        summaries.append({
            'seed': seed, 'method': method,
            'origin': 'new_revision_run' if method == NEW_METHOD else 'verified_historical_raw_reuse',
            'rmse': float(np.sqrt(np.mean([r['squared_state_error'] for r in rr]))),
            'mae_euclidean': float(np.mean([r['euclidean_state_error'] for r in rr])),
            'posterior_mean_rmse': float(np.sqrt(np.mean([r['squared_posterior_mean_error'] for r in rr]))),
            'mean_distinct_count': float(np.mean([r['distinct_count'] for r in rr])),
            'mean_subset_size': None if rr[0]['subset_size'] is None else float(np.mean([r['subset_size'] for r in rr])),
            'mean_generations': float(np.mean([r['generations'] for r in rr])),
            'mean_logfitness_gap': float(np.mean([r['optimal_logfitness']-r['logfitness'] for r in rr])),
            'mean_logfitness': float(np.mean([r['logfitness'] for r in rr])),
            'mean_ellipse_count': float(np.mean([r['ellipse_count'] for r in rr])),
            'runtime_s': float(sum(r['runtime_s'] for r in cc)),
            'runtime_scope': 'current initialization-inclusive operator time' if method == NEW_METHOD else 'historical batch; provenance only; no cross-batch speed ranking',
            'fitness_evaluations': sum(r['fitness_evaluations'] for r in rr),
            'likelihood_evaluations': sum(r['likelihood_evaluations'] for r in rr),
            'candidate_distance_evaluations': sum(r['candidate_distance_evaluations'] for r in rr),
            'binary_dimensions_processed': sum(r['binary_dimensions_processed'] for r in rr),
            'operation_counts': operations,
        })
    return summaries


def run_seed(seed, root, role, protocol, freeze, cfg):
    out = Path(root)/f'seed-{seed:02d}'; out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    environment = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
        'cpu': platform.processor(), 'device': 'CPU', 'dtype': 'float64', 'mixed_precision': False,
        'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'same_environment_scope_only': True}
    with io.StringIO() as fp:
        with redirect_stdout(fp):
            np.show_config()
        environment['numpy_build_config'] = fp.getvalue()
    save_json(out/'environment.json', environment)
    config = {'schema': 'EC.qzero_revision.v1', 'protocol': protocol, 'protocol_sha256': digest(PROTOCOL),
        'historical_freeze_sha256': digest(FREEZE), 'operator': asdict(cfg),
        'method': NEW_METHOD, 'label': 'E-qzero-consistent: self-consistent interpretation variant',
        'source_scope': 'One-step selection module; not original author intent, full tracker, or complete source-algorithm implementation',
        'only_algorithmic_change': 'best mask maps to alpha*(1-D)+(1-alpha)*D under q=P(bit0)',
        'algorithm_rng': 'numpy default_rng(SeedSequence(seed).spawn(7)[3]); same initial stream as archived literal; later draws can diverge through early stopping',
        'analysis_rng_seeds': {'bootstrap': 9301, 'sign_flip': 9302},
        'timing': 'perf_counter inside operator before shared-context construction, q/best initialization through summarization; excludes file IO and historical-reuse verification; timing is descriptive only',
        'measurement_boundary': 'bit1 if q<u; equality observes bit0',
        'counter_definitions': 'operation-count-definitions.json'}
    save_json(out/'config.json', config)
    snapshots = out/'code_snapshot'; snapshots.mkdir()
    for p in [BASE/'ecqpf_interpreted.py', BASE/'run_revision.py', BASE/'test_interpreted.py',
              BASE/'operation-count-definitions.json', HISTORICAL_OPERATOR,
              BASE/'test-red.log', BASE/'test-direction-red.log', BASE/'test-green.log']:
        shutil.copy2(p, snapshots/p.name)
    shutil.copy2(PROTOCOL, snapshots/PROTOCOL.name)
    save_json(out/'test_evidence.json', {'suite': 'test_interpreted.py', 'tests': 10, 'green_exit_code': 0,
        'red_missing_module': {'path': 'test-red.log', 'sha256': digest(BASE/'test-red.log')},
        'red_literal_direction_and_fixed_point': {'path': 'test-direction-red.log', 'sha256': digest(BASE/'test-direction-red.log'), 'failures': 3},
        'red_operator_sha256': digest(BASE/'ecqpf_interpreted.red.txt'),
        'green_log_sha256': digest(BASE/'test-green.log'),
        'scope': 'Meaningful gold/property/contract tests, not full-source algorithm validation'})
    rows = []; costs = []; histories = []; checks = []; new_estimates = []; new_indices = []
    status = 'failed'; input_hashes = []; aggregated = None
    try:
        source, input_hashes, truth, centers, clouds, oldrows, oldest, oldidx = verify_historical_seed(seed, freeze, cfg)
        for name in ['truth.npy', 'centers.npy', 'particles.npy']:
            shutil.copy2(source/name, out/name)
        for method in OLD_METHODS:
            for suffix in ['estimates', 'indices']:
                shutil.copy2(source/f'{method}_{suffix}.npy', out/f'{method}_{suffix}.npy')
        shutil.copy2(source/'raw_metrics.jsonl', out/'historical_raw_metrics.jsonl')
        shutil.copy2(source/'seed_summary.json', out/'historical_seed_summary.json')
        oldcosts = [r for r in json.loads((source/'costs.json').read_text(encoding='utf-8')) if r['method'] in OLD_METHODS]
        for r in oldrows:
            r = dict(r); r['origin'] = 'verified_historical_raw_reuse'
            r['subset_size'] = r['distinct_count'] if r['method'] != 'ellipse_systematic_fitness' else None
            r['operation_counts'] = historical_operations(r); rows.append(r)
        costs.extend([{**r, 'origin': 'historical_timing_not_cross_batch_rankable'} for r in oldcosts])
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(7)[3])
        for update in range(20):
            result = consistent_select(clouds[update], centers[update], cfg, rng)
            estimate = result['estimate']; new_estimates.append(estimate); new_indices.append(result['indices'])
            err2 = float(np.sum((estimate-truth[update])**2))
            posterior_err2 = float(np.sum((estimate-25./26.*centers[update])**2))
            row = {k: v for k, v in result.items() if k not in {'estimate', 'indices', 'ellipse_indices', 'runtime_s', 'generation_history'}}
            row.update({'seed': seed, 'update': update, 'method': NEW_METHOD, 'origin': 'new_revision_run',
                'squared_state_error': err2, 'euclidean_state_error': float(np.sqrt(err2)),
                'squared_posterior_mean_error': posterior_err2})
            rows.append(row); costs.append({'update': update, 'method': NEW_METHOD, 'runtime_s': result['runtime_s'], 'origin': 'new_revision_run'})
            histories.append({'update': update, 'columns': ['generation', 'best_logfitness', 'subset_size', 'mean_q_zero_probability'], 'history': result['generation_history']})
            ellipse, distances, variance = shared_context(clouds[update], centers[update], cfg)
            peers = [r for r in oldrows if r['update'] == update]
            same_context = all(r['ellipse_count'] == len(ellipse) and r['fitness_variance'] == variance and r['optimal_logfitness'] == result['optimal_logfitness'] for r in peers)
            all_inside = all(set(oldidx[m][update]).issubset(set(ellipse)) for m in OLD_METHODS)
            finite = np.isfinite(estimate).all() and np.isfinite(err2)
            bound = result['logfitness'] <= result['optimal_logfitness']+1e-12
            new_inside = set(result['indices']).issubset(set(ellipse))
            np.testing.assert_array_equal(result['ellipse_indices'], ellipse)
            np.testing.assert_allclose(estimate, clouds[update][result['indices']].mean(axis=0), rtol=0, atol=0)
            check = {'update': update, 'identical_shared_ellipse_variance_and_optimum': same_context,
                     'historical_indices_in_same_ellipse': all_inside, 'finite_estimate': bool(finite),
                     'fitness_bounded_by_exact': bool(bound), 'new_indices_in_same_ellipse': new_inside,
                     'fitness_count_contract': result['fitness_evaluations'] == 500*result['generations'],
                     'binary_count_contract': result['binary_dimensions_processed'] == 500*len(ellipse)*result['generations']}
            checks.append(check)
            if not all(v for k, v in check.items() if k != 'update'):
                raise AssertionError(f'EC shared-input/fitness/count guardrail failure at update {update}')
            # Save progress before proceeding: failures retain completed arrays/metrics.
            np.save(out/f'{NEW_METHOD}_estimates.npy', np.array(new_estimates))
            np.save(out/f'{NEW_METHOD}_indices.npy', np.array(new_indices))
            (out/'raw_metrics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows), encoding='utf-8')
            save_json(out/'costs.json', costs); save_json(out/'generation_history.json', histories)
        aggregated = summarize(seed, rows, costs); save_json(out/'seed_summary.json', aggregated)
        with (out/'predictions.csv').open('w', newline='', encoding='utf-8') as fp:
            names = ['seed', 'update', 'method', 'truth_x', 'truth_y', 'center_x', 'center_y', 'estimate_x', 'estimate_y']
            writer = csv.DictWriter(fp, fieldnames=names); writer.writeheader()
            for method in METHODS:
                estimates = np.array(new_estimates) if method == NEW_METHOD else oldest[method]
                for j in range(20):
                    writer.writerow(dict(zip(names, [seed, j, method, *truth[j], *centers[j], *estimates[j]])))
        save_json(out/'guardrails.json', {'status': 'PASS', 'truth_used_only_by_evaluator': True, 'checks': checks})
        status = 'completed'
        (out/'stdout.log').write_text(f'completed seed={seed}; updates=20; newly executed variants=1; verified reused comparators=3\n', encoding='utf-8')
        (out/'stderr.log').write_text('', encoding='utf-8')
    except Exception as exc:
        save_json(out/'failure.json', {'exception': type(exc).__name__, 'message': str(exc),
            'traceback': traceback.format_exc(), 'completed_new_updates': len(new_estimates),
            'completed_rows': rows, 'costs': costs, 'checks': checks})
        (out/'stdout.log').write_text(f'failed seed={seed}; retained completed evidence\n', encoding='utf-8')
        (out/'stderr.log').write_text(traceback.format_exc(), encoding='utf-8')
    def record(path):
        p = out/path
        return {'path': path, 'sha256': digest(p)} if p.exists() else None
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HISTORICAL, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=HISTORICAL, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {'schema': 'light.run_manifest.v3', 'run_id': f'EC-qzero__seed-{seed:02d}__{Path(root).name}',
        'matrix_row': 'EC-qzero-consistent', 'status': status,
        'termination': {'reason': 'natural_exit' if status == 'completed' else 'error', 'exit_code': 0 if status == 'completed' else 1, 'signal': None},
        'seed': {'role': role, 'value': seed}, 'started_at': started, 'ended_at': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, str(BASE/'run_revision.py'), '--seed', str(seed), '--out', str(root), '--role', role],
        'config': record('config.json'), 'environment': record('environment.json'),
        'code': {'commit': commit, 'dirty': bool(dirty), 'files': [{'path': str(p.relative_to(out)), 'sha256': digest(p)} for p in snapshots.iterdir()]},
        'inputs': input_hashes, 'historical_freeze_sha256': digest(FREEZE),
        'artifacts': {k: record(v) for k, v in {'stdout': 'stdout.log', 'stderr': 'stderr.log', 'raw_metrics': 'raw_metrics.jsonl',
            'predictions': 'predictions.csv', 'test_evidence': 'test_evidence.json', 'guardrail_evidence': 'guardrails.json', 'failure': 'failure.json'}.items()},
        'completion': {'status': 'PASS' if status == 'completed' else 'FAIL',
            'oracle': ['ten operator tests passed', 'all20 updates retained', 'shared ellipse/V/input and exact-fitness bounds', 'historical hashes and metrics verified'],
            'evidence_artifacts': ['test_evidence', 'raw_metrics', 'predictions', 'guardrail_evidence']},
        'reproducibility': {'pair_id': 'EC-qzero__seed-0' if role == 'fixed_repro' else None,
            'comparison_role': Path(root).name, 'compare_artifacts': ['predictions', 'raw_metrics']},
        'notes': ['self-consistent interpretation; not original author intent or complete source algorithm',
            'descriptive runtime only; no speed ranking against reused historical timing']}
    manifest['output_files'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}
    save_json(out/'manifest.json', manifest)
    if status != 'completed':
        raise RuntimeError(f'dataset {seed} failed; retained {out}/failure.json')
    print(json.dumps({'seed': seed, 'status': status, 'consistent_rmse': aggregated[0]['rmse'], 'new_operator_wall_s': aggregated[0]['runtime_s']}), flush=True)
    return aggregated


def main():
    p = argparse.ArgumentParser(); p.add_argument('--seed', type=int); p.add_argument('--out', required=True)
    p.add_argument('--role', choices=['randomness_estimation', 'fixed_repro'], default='randomness_estimation')
    args = p.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0' or os.environ.get('OPENBLAS_NUM_THREADS') != '1' or os.environ.get('OMP_NUM_THREADS') != '1':
        raise RuntimeError('Launch a new process with PYTHONHASHSEED=0, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1')
    protocol, freeze, cfg = validate_protocol()
    seeds = protocol['EC']['seeds'] if args.seed is None else [args.seed]
    if any(seed not in protocol['EC']['seeds'] for seed in seeds):
        raise ValueError('requested seed is outside the frozen matrix')
    root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    save_json(root/'batch_config.json', {'protocol_sha256': digest(PROTOCOL), 'seeds': seeds, 'role': args.role,
        'analysis_rng_seeds': {'bootstrap': 9301, 'sign_flip': 9302}, 'max_wall_s': 900, 'historical_raw_reuse': OLD_METHODS})
    begin = time.perf_counter(); summaries = []
    try:
        for seed in seeds:
            if time.perf_counter()-begin > 900:
                raise TimeoutError('frozen local EC wall budget of900 seconds reached')
            summaries += run_seed(seed, root, args.role, protocol, freeze, cfg)
        save_json(root/'summary.json', summaries)
        save_json(root/'batch_manifest.json', {'status': 'completed', 'completed_seeds': seeds,
            'all_seeds_reported': True, 'wall_s_including_IO_and_reuse_verification': time.perf_counter()-begin,
            'summary_sha256': digest(root/'summary.json'), 'protocol_sha256': digest(PROTOCOL)})
    except Exception as exc:
        save_json(root/'batch_failure.json', {'exception': type(exc).__name__, 'message': str(exc),
            'traceback': traceback.format_exc(), 'completed_summary_rows': summaries,
            'no_seed_exclusions_or_replacements': True})
        raise


if __name__ == '__main__':
    main()
