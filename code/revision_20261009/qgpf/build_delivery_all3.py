"""Build a compact scientific/timing handoff without changing run identities."""
from pathlib import Path
import json,csv,statistics
import numpy as np
from run_revision import HERE,sha,dump,SCIENTIFIC_ARRAYS

def main():
    source=HERE/'analysis-qgpf-all3.json'
    a=json.loads(source.read_text(encoding='utf-8'))
    files=['qgpf_stable.py','run_revision.py','test_stable.py','analyze_revision.py','export_publication_tables.py',
           'analysis-qgpf-all3.json','analysis-qgpf-all3.md','recalculated-qgpf-all3.csv',
           'publication-descriptive.csv','publication-counts.csv','publication-diagnostics.csv','publication-family3.csv',
           'publication-table-manifest.json','runs-m1/batch_manifest.json','runs-s1/batch_manifest.json',
           'runs-m1-fixed-replay/fixed_seed_replay.json','runs-s1-fixed-replay/fixed_seed_replay.json',
           'batch-m1-execution.log','batch-s1-execution.log','operation-count-definitions.json',
           'public-log-conversion.json','test-red-public.log','anonymize_test_log.py','README.md','.gitignore']
    evidence={'schema':'revision.qgpf.delivery.v1','status':'SCIENTIFIC_MATRIX_COMPLETE',
      'completed':{'B-stable':30,'Q-stable':30,'Q-stable-retain':30,'failures':0,'seed_exclusions':0},
      'fixed_seed_independent_process_replays':{'M1':True,'retention':True},
      'primary_summary':{m:s['rmse_position'] for m,s in a['summaries'].items()},
      'comparisons':a['comparisons'],'holm_status':a['holm_status'],
      'recalculation_runs':len(a['recalculation_checks']),'files_sha256':{p:sha(HERE/p) for p in files},
      'scope':a['scope'],'timing_status':'pending separate reserved single-load samples',
      'limitations':['Numerical and source-retention corrections demonstrate no primary benefit within the paired intervals; this is not evidence of equivalence',
        'Coordinate position RMSE is not Euclidean RMSE','No speed inference from parallel logs or historical timers',
        'Counts overlap as defined; no physical quantum gate claims',
        'Stable-versus-floor diagnostic applies only to ratio Q-stable; initialized zeros in other variants are not agreement evidence',
        'Canonical Light checkpoint not executed; no canonical stage PASS claimed'],
      'public_exclusions':['test-red.log','__pycache__/','*.pyc']}
    timing_root=HERE/'runs-serial-timing'
    if (timing_root/'batch_manifest.json').exists():
        timer_rows=[];checks=[]
        for method in ('B-stable','Q-stable','Q-stable-retain'):
            original=HERE/('runs-s1' if method=='Q-stable-retain' else 'runs-m1')
            for seed in range(3):
                run=timing_root/method/f'seed-{seed:02d}';old=original/method/f'seed-{seed:02d}'
                manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'));assert manifest['status']=='completed'
                for name,expected in manifest['files_sha256'].items():assert sha(run/name)==expected
                equal={name:bool(np.array_equal(np.load(run/f'{name}.npy',allow_pickle=False),np.load(old/f'{name}.npy',allow_pickle=False))) for name in SCIENTIFIC_ARRAYS}
                m=json.loads((run/'raw_metrics.json').read_text(encoding='utf-8'));om=json.loads((old/'raw_metrics.json').read_text(encoding='utf-8'))
                assert all(equal.values()) and all(m[key]==om[key] for key in m if key not in ('runtime_s','initialization_runtime_s'))
                assert sha(run/'rng_final_state.json')==sha(old/'rng_final_state.json')
                env=json.loads((run/'environment.json').read_text(encoding='utf-8'))
                assert env['scheduler']['max_independent_processes']==1
                assert all(env['threads'][key]=='1' for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
                timer_rows.append({'method':method,'seed':seed,'runtime_s':m['runtime_s'],'initialization_runtime_s':m['initialization_runtime_s'],
                 'cpu':env['cpu'],'pid':env['pid'],'scope':'initialization + filtering + diagnostics; excluding I/O and evaluation metrics'})
                checks.append({'method':method,'seed':seed,'scientific_arrays_equal':equal,'non_timer_metrics_and_rng_equal':True,
                  'file_hashes_passed':True,'one_worker_one_BLAS_thread':True})
        summaries={}
        for method in ('B-stable','Q-stable','Q-stable-retain'):
            values=[row['runtime_s'] for row in timer_rows if row['method']==method]
            summaries[method]={'n':3,'mean_s':statistics.mean(values),'sd_s':statistics.stdev(values),
                              'median_s':statistics.median(values),'min_s':min(values),'max_s':max(values)}
        with (HERE/'serial-timing-seeds0-2.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(timer_rows[0]));w.writeheader();w.writerows(timer_rows)
        dump(HERE/'serial-timing-seeds0-2.json',{'schema':'revision.qgpf.serial-timing.v1','rows':timer_rows,
          'summaries':summaries,'checks':checks,'sample_selection':'Reserved seeds 0–2; not additional independent scientific samples',
          'inference':'Three single-load samples per new variant; descriptive instrumented CPU time only; no historical or physical quantum speed comparison',
          'batch_sha256':sha(timing_root/'batch_manifest.json')})
        evidence['timing_status']='COMPLETE_RESERVED_SEEDS_0_2_SINGLE_LOAD'
        for name in ('serial-timing-seeds0-2.csv','serial-timing-seeds0-2.json','batch-serial-timing-execution.log'):
            evidence['files_sha256'][name]=sha(HERE/name)
        evidence['timing_summaries']=summaries
    dump(HERE/'delivery-qgpf-all3.json',evidence)
    print(json.dumps({'status':evidence['status'],'timing_status':evidence['timing_status']},indent=2))
if __name__=='__main__':main()
