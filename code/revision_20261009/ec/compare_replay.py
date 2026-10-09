"""Compare independent EC fixed-seed scientific outputs, excluding timing."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(first, second, out):
    first = Path(first); second = Path(second)
    names = ['truth.npy', 'centers.npy', 'particles.npy', 'ec_qzero_consistent_estimates.npy',
             'ec_qzero_consistent_indices.npy', 'raw_metrics.jsonl', 'generation_history.json',
             'guardrails.json', 'predictions.csv']
    records = []
    for name in names:
        ha = digest(first/name); hb = digest(second/name)
        records.append({'path': name, 'first_sha256': ha, 'second_sha256': hb, 'equal': ha == hb})
    for folder in [first, second]:
        environment = json.loads((folder/'environment.json').read_text(encoding='utf-8'))
        if environment['OPENBLAS_NUM_THREADS'] != '1' or environment['OMP_NUM_THREADS'] != '1' or environment['PYTHONHASHSEED'] != '0':
            raise AssertionError('fixed environment flags were not applied')
    ea = json.loads((first/'environment.json').read_text(encoding='utf-8'))
    eb = json.loads((second/'environment.json').read_text(encoding='utf-8'))
    same_env = ea == eb
    scientific_match = same_env and all(r['equal'] for r in records)
    result = {'schema': 'EC.same_environment_replay.v1', 'status': 'PASS' if scientific_match else 'FAIL',
        'same_environment': same_env, 'first': str(first), 'second': str(second), 'records': records,
        'excluded': ['costs.json', 'seed_summary.json runtime fields', 'timestamps and provenance paths'],
        'scope': 'Two independent directories/processes in this environment; no claim of cross-hardware reproducibility'}
    Path(out).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if not scientific_match:
        raise AssertionError('independent EC scientific replay mismatch')
    print('PASS: independent fixed-seed scientific outputs agree; runtime excluded')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--first', required=True); p.add_argument('--second', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); compare(a.first, a.second, a.out)
