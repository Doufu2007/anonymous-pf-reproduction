"""Analyze completed EC q-zero batch at the frozen dataset level.

The script never ranks runtime across the newly executed and reused batches.
It verifies one RMSE per seed and method, then emits paired mean differences,
percentile bootstrap CIs, sign-flip p-values, dz, and Holm family-3 results.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

BASE = Path(__file__).resolve().parent
PROTOCOL = BASE.parents[3] / 'revision-2026-10-09' / 'frozen-protocol.json'
METHODS = ['ec_qzero_consistent', 'source_ecqpf_operator', 'ellipse_exact_fitness', 'ellipse_systematic_fitness']
COMPARISONS = [('consistent_minus_literal', METHODS[0], METHODS[1]),
               ('consistent_minus_exact', METHODS[0], METHODS[2]),
               ('consistent_minus_systematic', METHODS[0], METHODS[3])]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def holm(pvalues):
    order = np.argsort(np.asarray(pvalues, dtype=float), kind='mergesort')
    adjusted = np.empty(len(order), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order)-rank)*float(pvalues[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {'mean': float(values.mean()), 'sd': float(values.std(ddof=1)),
            'min': float(values.min()), 'max': float(values.max()),
            'median': float(np.median(values)),
            'q25': float(np.quantile(values, .25)), 'q75': float(np.quantile(values, .75))}


def analyze(batch_root, out_root):
    batch_root = Path(batch_root); out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8-sig'))
    if protocol['EC']['statistics']['bootstrap'] != 20000 or protocol['EC']['statistics']['sign_flip'] != 50000:
        raise AssertionError('analysis counts differ from frozen protocol')
    seeds = protocol['EC']['seeds']
    summaries = {}; raw_hashes = {}
    for seed in seeds:
        path = batch_root/f'seed-{seed:02d}'/'seed_summary.json'
        if not path.exists():
            raise FileNotFoundError(path)
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
        if {r['method'] for r in rows} != set(METHODS):
            raise AssertionError(f'seed {seed} missing methods')
        if any(r['seed'] != seed for r in rows):
            raise AssertionError(f'seed {seed} summary identity mismatch')
        summaries[seed] = {r['method']: r for r in rows}
        raw = batch_root/f'seed-{seed:02d}'/'raw_metrics.jsonl'
        if not raw.exists():
            raise FileNotFoundError(raw)
        records = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines() if line.strip()]
        truth_path = path.parent/'truth.npy'
        truth = np.load(truth_path, allow_pickle=False)
        clouds = np.load(path.parent/'particles.npy', allow_pickle=False)
        if truth.shape != (20, 2) or clouds.shape != (20, 500, 2):
            raise AssertionError('archived input shape mismatch')
        if len(records) != 80 or len({(r['method'], r['update']) for r in records}) != 80:
            raise AssertionError(f'seed {seed} must contain 4 methods x20 updates')
        for method in METHODS:
            rr = [r for r in records if r['method'] == method]
            if len(rr) != 20 or any(r['seed'] != seed for r in rr):
                raise AssertionError(f'seed {seed}/{method} has incomplete raw rows')
            recomputed = float(np.sqrt(np.mean([r['squared_state_error'] for r in rr])))
            estimates_path = path.parent/f'{method}_estimates.npy'
            indices_path = path.parent/f'{method}_indices.npy'
            estimates = np.load(estimates_path, allow_pickle=False)
            indices = np.load(indices_path, allow_pickle=False)
            if estimates.shape != (20, 2) or indices.shape != (20, 500):
                raise AssertionError('returned arrays must retain all20 updates')
            np.testing.assert_allclose(np.stack([clouds[j][indices[j]].mean(axis=0) for j in range(20)]), estimates, rtol=0, atol=1e-14)
            array_rmse = float(np.sqrt(np.mean(np.sum((estimates-truth)**2, axis=1))))
            np.testing.assert_allclose(array_rmse, recomputed, rtol=0, atol=1e-14)
            reported = summaries[seed][method]['rmse']
            if not np.isclose(recomputed, reported, rtol=0, atol=1e-14):
                raise AssertionError(f'RMSE mismatch for seed {seed}/{method}')
        raw_hashes[seed] = {'summary': digest(path), 'raw_metrics': digest(raw), 'truth': digest(truth_path),
            'arrays': {m: {'estimates': digest(path.parent/f'{m}_estimates.npy'), 'indices': digest(path.parent/f'{m}_indices.npy')} for m in METHODS}}
    matrix = {method: np.array([summaries[s][method]['rmse'] for s in seeds], dtype=np.float64) for method in METHODS}
    if not all(np.isfinite(v).all() for v in matrix.values()):
        raise AssertionError('nonfinite RMSE in completed batch')
    bootstrap_rng = np.random.default_rng(9301)
    bootstrap_indices = bootstrap_rng.integers(0, len(seeds), size=(20000, len(seeds)))
    sign_rng = np.random.default_rng(9302)
    sign_flips = sign_rng.choice(np.array([-1.0, 1.0]), size=(50000, len(seeds)))
    method_summary = {method: summarize(matrix[method]) for method in METHODS}
    secondary_summary = {}
    secondary_names = ['mean_distinct_count', 'mean_subset_size', 'mean_generations', 'mean_logfitness_gap',
                       'mean_logfitness', 'mean_ellipse_count', 'fitness_evaluations',
                       'likelihood_evaluations', 'candidate_distance_evaluations', 'binary_dimensions_processed']
    for method in METHODS:
        item = {}
        for endpoint in secondary_names:
            vals = [summaries[s][method][endpoint] for s in seeds]
            item[endpoint] = None if any(v is None for v in vals) else summarize(vals)
        counter_names = summaries[seeds[0]][method]['operation_counts']
        item['operation_counts'] = {}
        for name in counter_names:
            vals = [summaries[s][method]['operation_counts'][name] for s in seeds]
            item['operation_counts'][name] = None if any(v is None for v in vals) else summarize(vals)
        secondary_summary[method] = item
    runtime_descriptive = {method: {'origin': 'current_revision' if method == METHODS[0] else 'historical_raw_reuse',
        'summary_total_operator_seconds_per_dataset': summarize([summaries[s][method]['runtime_s'] for s in seeds]),
        'no_cross_batch_speed_rank': True} for method in METHODS}
    comparison_rows = []; paired_differences = {}
    for name, left, right in COMPARISONS:
        diff = matrix[left] - matrix[right]; paired_differences[name] = diff
        observed = float(diff.mean())
        boot_means = diff[bootstrap_indices].mean(axis=1)
        ci = np.percentile(boot_means, [2.5, 97.5])
        signed_means = (sign_flips*diff).mean(axis=1)
        mc_p = float((np.sum(np.abs(signed_means) >= abs(observed)) + 1) / (len(signed_means)+1))
        dz = float(observed/diff.std(ddof=1)) if diff.std(ddof=1) else None
        half = float(student_t.ppf(.975, len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff)))
        comparison_rows.append({'comparison': name, 'left': left, 'right': right, 'n_datasets': len(seeds),
            'mean_difference_rmse': observed, 'sd_difference_rmse': float(diff.std(ddof=1)),
            'bootstrap_resamples': len(boot_means), 'bootstrap_seed': 9301,
            'bootstrap_ci95_low': float(ci[0]), 'bootstrap_ci95_high': float(ci[1]),
            'signflip_resamples': len(signed_means), 'signflip_seed': 9302,
            'signflip_mc_p_raw': mc_p, 'paired_dz': dz,
            'paired_t_ci95_low': observed-half, 'paired_t_ci95_high': observed+half,
            't_interval_role': 'descriptive sensitivity alongside the frozen primary dataset-bootstrap interval; not a new tested contrast',
            'interpretation': 'paired dataset RMSE difference; not 600 independent updates'})
    adjusted = holm([row['signflip_mc_p_raw'] for row in comparison_rows])
    for row, p in zip(comparison_rows, adjusted):
        row['holm_family'] = 3; row['signflip_mc_p_holm3'] = float(p)
    with (out_root/'paired_rmse.csv').open('w', newline='', encoding='utf-8') as fp:
        fields = ['seed'] + METHODS
        writer = csv.DictWriter(fp, fieldnames=fields); writer.writeheader()
        for i, seed in enumerate(seeds):
            writer.writerow({'seed': seed, **{m: matrix[m][i] for m in METHODS}})
    with (out_root/'paired_differences.csv').open('w', newline='', encoding='utf-8') as fp:
        fields = ['seed'] + [c[0] for c in COMPARISONS]
        writer = csv.DictWriter(fp, fieldnames=fields); writer.writeheader()
        for i, seed in enumerate(seeds):
            writer.writerow({'seed': seed, **{name: paired_differences[name][i] for name, _, _ in COMPARISONS}})
    result = {'schema': 'EC.qzero_revision.analysis.v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': digest(PROTOCOL), 'batch_root': str(batch_root), 'n_datasets': len(seeds),
        'updates_per_dataset': 20, 'unit': 'dataset-level RMSE; not updates as independent observations',
        'method_summary': method_summary, 'comparisons': comparison_rows,
        'secondary_summary': secondary_summary, 'runtime_descriptive': runtime_descriptive,
        'raw_hashes_by_seed': raw_hashes,
        'runtime': 'Excluded from all scientific identity and paired RMSE comparisons; reused historical runtime retained for provenance only',
        'scope': 'Exploratory revision on previously observed archived synthetic inputs; consistent is a self-consistent q-zero interpretation variant, not original author intent or complete source algorithm'}
    save_json(out_root/'analysis.json', result)
    lines = ['# EC q-zero interpretation analysis', '',
             'Unit: 30 datasets, each aggregating 20 updates into one 2D vector RMSE.', '',
             '| Method | Mean RMSE | SD | Min | Max |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        s = method_summary[method]; lines.append(f"| {method} | {s['mean']:.8g} | {s['sd']:.8g} | {s['min']:.8g} | {s['max']:.8g} |")
    lines += ['', '| Comparison | Mean diff | Bootstrap 95% CI | Sign-flip p | Holm-3 p | dz |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in comparison_rows:
        lines.append(f"| {row['comparison']} | {row['mean_difference_rmse']:.8g} | [{row['bootstrap_ci95_low']:.8g}, {row['bootstrap_ci95_high']:.8g}] | {row['signflip_mc_p_raw']:.8g} | {row['signflip_mc_p_holm3']:.8g} | {row['paired_dz'] if row['paired_dz'] is not None else 'NA'} |")
    lines += ['', 'Bootstrap resamples whole datasets; sign-flip Monte Carlo resolution is1/50001. Holm family is the three frozen comparisons.',
              'Secondaries and operation counts are descriptive and retained in analysis.json. No secondary hypothesis tests were added.',
              'Historical comparator runtime is retained only as provenance; this revision does not make a cross-batch speed claim.',
              'Consistent is a self-consistent q-zero interpretation variant, not original author intent or a complete source tracker.']
    (out_root/'analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save_json(out_root/'analysis_manifest.json', {'status': 'completed', 'protocol_sha256': digest(PROTOCOL),
        'input_batch': str(batch_root), 'input_summary_hashes': raw_hashes,
        'outputs': {p.name: digest(p) for p in out_root.iterdir() if p.is_file()}})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--batch', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); analyze(args.batch, args.out)


if __name__ == '__main__':
    main()
