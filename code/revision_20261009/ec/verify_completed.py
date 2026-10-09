"""Read-only independent checksum and array/fitness audit of EC revision runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from run_revision import (OLD_METHODS, NEW_METHOD, digest, save_json,
    validate_protocol, INPUT_ROOT, WORKSPACE, shared_context, json_rows)


def verify(root, out):
    root = Path(root); protocol, freeze, cfg = validate_protocol()
    batch = json.loads((root/'batch_manifest.json').read_text(encoding='utf-8'))
    if batch['status'] != 'completed' or batch['completed_seeds'] != protocol['EC']['seeds']:
        raise AssertionError('the30-dataset matrix is incomplete')
    if digest(root/'summary.json') != batch['summary_sha256']:
        raise AssertionError('batch summary checksum mismatch')
    records = []; file_count = 0; input_records = 0; new_rows = 0; reused_rows = 0
    for seed in protocol['EC']['seeds']:
        folder = root/f'seed-{seed:02d}'
        historical = INPUT_ROOT/f'seed-{seed:02d}'
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest['status'] != 'completed' or manifest['completion']['status'] != 'PASS' or manifest['seed']['value'] != seed:
            raise AssertionError(f'seed {seed} manifest completion failed')
        for rel, checksum in manifest['output_files'].items():
            if digest(folder/rel) != checksum:
                raise AssertionError(f'seed {seed} output checksum mismatch: {rel}')
            file_count += 1
        for item in manifest['inputs']:
            if digest(WORKSPACE/item['path']) != item['sha256'] or freeze['inputs'][item['path']] != item['sha256']:
                raise AssertionError('archived provenance checksum mismatch')
            input_records += 1
        guardrails = json.loads((folder/'guardrails.json').read_text(encoding='utf-8'))
        if guardrails['status'] != 'PASS' or len(guardrails['checks']) != 20:
            raise AssertionError(f'seed {seed} guardrails incomplete')
        for check in guardrails['checks']:
            if not all(val for key, val in check.items() if key != 'update'):
                raise AssertionError(f'seed {seed} false guardrail')
        truth = np.load(folder/'truth.npy', allow_pickle=False)
        centers = np.load(folder/'centers.npy', allow_pickle=False)
        clouds = np.load(folder/'particles.npy', allow_pickle=False)
        for name in ['truth.npy', 'centers.npy', 'particles.npy']:
            if digest(folder/name) != digest(historical/name):
                raise AssertionError(f'seed {seed} copied input mismatch')
        allrows = json_rows(folder/'raw_metrics.jsonl')
        if len(allrows) != 80 or len({(r['method'], r['update']) for r in allrows}) != 80:
            raise AssertionError('expected4 methods x20 rows with unique keys')
        summaries = {r['method']: r for r in json.loads((folder/'seed_summary.json').read_text(encoding='utf-8'))}
        for method in [NEW_METHOD] + OLD_METHODS:
            estimates = np.load(folder/f'{method}_estimates.npy', allow_pickle=False)
            indices = np.load(folder/f'{method}_indices.npy', allow_pickle=False)
            if estimates.shape != (20, 2) or indices.shape != (20, 500) or indices.dtype.kind not in 'iu':
                raise AssertionError('output shape/dtype contract mismatch')
            if np.any((indices < 0) | (indices >= 500)) or not np.isfinite(estimates).all():
                raise AssertionError('invalid returned indices/estimate')
            rows = {r['update']: r for r in allrows if r['method'] == method}
            for j in range(20):
                row = rows[j]
                ellipse, distances, variance = shared_context(clouds[j], centers[j], cfg)
                if not set(indices[j]).issubset(set(ellipse)):
                    raise AssertionError('returned particle outside common ellipse')
                np.testing.assert_allclose(estimates[j], clouds[j][indices[j]].mean(axis=0), rtol=0, atol=1e-14)
                unique = np.unique(indices[j]); distinct = len(unique)
                actual_fitness = -float(np.linalg.norm(clouds[j][unique]-centers[j], axis=1).mean())/(2*variance)
                actual_optimum = -float(distances.min())/(2*variance)
                np.testing.assert_allclose(row['logfitness'], actual_fitness, rtol=0, atol=1e-14)
                np.testing.assert_allclose(row['optimal_logfitness'], actual_optimum, rtol=0, atol=1e-14)
                np.testing.assert_allclose(row['fitness_variance'], variance, rtol=0, atol=0)
                if row['logfitness'] > actual_optimum+1e-12 or row['distinct_count'] != distinct or row['ellipse_count'] != len(ellipse):
                    raise AssertionError('fitness bound or subset count violated')
                np.testing.assert_allclose(row['squared_state_error'], np.sum((estimates[j]-truth[j])**2), rtol=0, atol=1e-12)
                if method == NEW_METHOD:
                    if row['fitness_evaluations'] != 500*row['generations'] or row['binary_dimensions_processed'] != 500*len(ellipse)*row['generations']:
                        raise AssertionError('EC generation/binary/fitness count mismatch')
                    if row['subset_size'] != distinct or row['operation_counts']['probability_update_entries'] != 500*len(ellipse)*row['probability_update_steps']:
                        raise AssertionError('EC subset/update count mismatch')
                    if row['likelihood_evaluations'] != 0 or row['operation_counts']['rotation_events'] != 0 or row['operation_counts']['crossover_attempts'] != 0:
                        raise AssertionError('nonapplicable EC operation must be zero')
                    new_rows += 1
                else:
                    reused_rows += 1
            recomputed_rmse = float(np.sqrt(np.mean(np.sum((estimates-truth)**2, axis=1))))
            np.testing.assert_allclose(summaries[method]['rmse'], recomputed_rmse, rtol=0, atol=1e-14)
            if method in OLD_METHODS:
                for suffix in ['estimates', 'indices']:
                    if digest(folder/f'{method}_{suffix}.npy') != digest(historical/f'{method}_{suffix}.npy'):
                        raise AssertionError('reused historical array checksum mismatch')
        histories = json.loads((folder/'generation_history.json').read_text(encoding='utf-8'))
        if len(histories) != 20:
            raise AssertionError('generation histories incomplete')
        for item in histories:
            history = np.array(item['history'], dtype=np.float64)
            row = next(r for r in allrows if r['method'] == NEW_METHOD and r['update'] == item['update'])
            if len(history) != row['generations'] or np.any(np.diff(history[:, 1]) < 0):
                raise AssertionError('retained global-best history must be monotonic')
            np.testing.assert_allclose(history[-1, 1], row['logfitness'], rtol=0, atol=1e-14)
        records.append({'seed': seed, 'status': 'PASS', 'manifest_sha256': digest(folder/'manifest.json'),
                        'raw_metrics_sha256': digest(folder/'raw_metrics.jsonl')})
    result = {'schema': 'EC.revision.readonly_verification.v1', 'status': 'PASS',
        'datasets': records, 'n_datasets': 30, 'new_updates_verified': new_rows,
        'reused_update_rows_verified': reused_rows, 'manifest_output_files_checked': file_count,
        'historical_input_records_checked': input_records,
        'checks': ['all manifest output-file hashes', 'historical provenance hashes', 'identical copied arrays',
                   'actual returned ensemble means', 'truth-only RMSE recalculation', 'all fitness values and exact upper bound',
                   'eligible ellipse/V equality', 'operation count contracts', 'monotonic global-best histories'],
        'scientific_unit': '30 datasets, not600 updates', 'no_failures_or_exclusions': True}
    save_json(out, result)
    print(json.dumps({'status': 'PASS', 'new_updates_verified': new_rows, 'output_files_checked': file_count}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--batch', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); verify(args.batch, args.out)
