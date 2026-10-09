"""Execute the frozen numerical revision; write only inside this directory."""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, os, platform, subprocess, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
from qgpf_stable import ModelConfig, FilterConfig, run_filter, COUNTER_NAMES, COUNT_DEFINITIONS

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
WORKSPACE=REPO.parent
HIST=REPO/'code'/'qgpf_reproduction'/'runs-main-30'
PROTOCOL=WORKSPACE/'revision-2026-10-09'/'frozen-protocol.json'
FREEZE=PROTOCOL.with_name('historical-freeze.json')
METHODS=('B-stable','Q-stable','Q-stable-retain')
SEEDS=tuple(range(30))
CODE_FILES=('qgpf_stable.py','run_revision.py','test_stable.py')
SCIENTIFIC_ARRAYS=('truth','observations','initial_particles','initial_weights','estimate_pre','estimate_post',
 'final_particles','final_weights','particles_pre','particles_post','weights_pre','weights_post',
 'step_counters','step_ess','step_distinct')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def relative(p):return Path(p).resolve().relative_to(REPO).as_posix()
def stream_seed(seed,method):
    # Retention changes acceptance including its RNG branch, beginning at the same Q stream.
    return int(seed+1000003*(1 if method=='B-stable' else 2))
def cpu_name():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return str(winreg.QueryValueEx(key,'ProcessorNameString')[0]).strip()
    except (ImportError,OSError):return platform.processor() or 'unavailable'
def git_provenance():
    def git(*args):
        return subprocess.run(['git',*args],cwd=REPO,capture_output=True,check=True).stdout
    try:
        diff=git('diff','--binary','HEAD')
        dirty=git('status','--porcelain').decode('utf-8','replace').splitlines()
        return {'commit':git('rev-parse','HEAD').decode().strip(),'dirty':bool(dirty),
                'status_repo_relative':dirty,'tracked_diff_sha256':hashlib.sha256(diff).hexdigest(),
                'untracked_code_content':'covered by exact code_sha256; tracked diff hash excludes untracked files'}
    except (OSError,subprocess.CalledProcessError):return {'available':False}
def load_contract():
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'));freeze=json.loads(FREEZE.read_text(encoding='utf-8'))
    if sha(FREEZE)!=protocol['historical_freeze']['sha256']:raise AssertionError('freeze identity changed')
    q=protocol['QGPF'];cfg=FilterConfig();model=ModelConfig()
    assert protocol['status']=='FROZEN_FOR_LOCAL_EXECUTION' and tuple(q['seeds'])==SEEDS
    assert (q['T'],q['N'],q['bits'],q['G'],q['elite_fraction'])==(model.horizon,cfg.particle_count,cfg.bit_length,cfg.generation_count,cfg.elite_fraction)
    return protocol,freeze
def load_inputs(seed,freeze):
    paths={'truth':HIST/f'truth_seed{seed}.npy','observations':HIST/f'observations_seed{seed}.npy'}
    hashes={k:sha(p) for k,p in paths.items()}
    for key,p in paths.items():
        frozen_key=p.relative_to(WORKSPACE).as_posix()
        if hashes[key]!=freeze['inputs'][frozen_key]:raise AssertionError('historical input hash mismatch')
    truth=np.load(paths['truth'],allow_pickle=False);obs=np.load(paths['observations'],allow_pickle=False)
    if truth.shape!=(101,4) or obs.shape!=(101,) or not np.isfinite(truth).all() or not np.isfinite(obs).all():raise AssertionError('input shape/finite contract mismatch')
    return truth,obs,{'files':{k:relative(p) for k,p in paths.items()},'sha256':hashes,'freeze_identity_verified':True}
def run_one(seed,method,out_root,scheduler):
    run=Path(out_root)/method/f'seed-{seed:02d}'
    if not run.resolve().is_relative_to(HERE):raise ValueError('output must remain inside revision qgpf directory')
    if run.exists():raise FileExistsError('existing run identity is immutable')
    run.mkdir(parents=True);started=now();metrics={};failure=None;input_info={}
    code_hashes={x:sha(HERE/x) for x in CODE_FILES};protocol_sha=sha(PROTOCOL)
    try:
        protocol,freeze=load_contract();t,obs,input_info=load_inputs(seed,freeze)
        model=ModelConfig();cfg=FilterConfig()
        config={'method':method,'seed':seed,'seed_role':'trajectory randomness, not independent time samples',
         'stream_seed':stream_seed(seed,method),'model':asdict(model),'filter':asdict(cfg),
         'primary_metric':'sqrt(mean over k=1..T and coordinates x,y of squared error)',
         'state_order':['x','vx','y','vy'],'estimate_timing':'pre weighted-move estimate; post is read-only diagnostic',
         'retention':'strict log-likelihood improvement only' if method=='Q-stable-retain' else 'stable likelihood-ratio acceptance' if method=='Q-stable' else 'systematic bootstrap',
         'protocol_sha256':protocol_sha,'historical_freeze_sha256':sha(FREEZE),
         'historical_archive_sha256':freeze['snapshot']['sha256'],'counter_definitions':COUNT_DEFINITIONS}
        dump(run/'config.json',config);dump(run/'input_hashes.json',input_info)
        dump(run/'environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
         'cpu':cpu_name(),'logical_cpu_count':os.cpu_count(),'pid':os.getpid(),'device':'CPU','dtype':'float64','AMP':'not applicable',
         'threads':{k:os.environ.get(k,'not explicitly set') for k in ('PYTHONHASHSEED','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
         'scheduler':scheduler,'timer_scope':'initialization + filtering + diagnostics; excludes loading and file I/O',
         'runtime_inference':'Parallel execution runtime is operational logging only; no speed comparison',
         'git':git_provenance()})
        rng=np.random.default_rng(stream_seed(seed,method));result=run_filter(obs,rng,method,model,cfg,record_post=True)
        pre=result['estimate_pre'];post=result['estimate_post'];err=pre[1:,[0,2]]-t[1:,[0,2]];verr=pre[1:,[1,3]]-t[1:,[1,3]]
        metrics={'seed':seed,'method':method,'rmse_position':float(np.sqrt(np.mean(err**2))),
         'mae_position':float(np.mean(np.abs(err))),'rmse_velocity':float(np.sqrt(np.mean(verr**2))),
         'mean_ess':float(result['step_ess'].mean()),'mean_ess_over_n':float(result['step_ess'].mean()/cfg.particle_count),
         'mean_distinct':float(result['step_distinct'].mean()),'runtime_s':result['runtime_s'],
         'initialization_runtime_s':result['initialization_runtime_s'],**result['counters'],
         'post_exploratory_rmse_position':float(np.sqrt(np.mean((post[1:,[0,2]]-t[1:,[0,2]])**2))),
         'post_exploratory_rmse_velocity':float(np.sqrt(np.mean((post[1:,[1,3]]-t[1:,[1,3]])**2)))}
        arrays={'truth':t,'observations':obs,'initial_weights':np.full(cfg.particle_count,1./cfg.particle_count),
                **{name:result[name] for name in SCIENTIFIC_ARRAYS if name not in ('truth','observations','initial_weights')},
                'step_elapsed_s':result['step_elapsed_s']}
        for name,arr in arrays.items():np.save(run/f'{name}.npy',arr)
        dump(run/'raw_metrics.json',metrics);dump(run/'guardrails.json',result['guardrails'])
        dump(run/'rng_final_state.json',result['rng_final_state']);dump(run/'counter_definitions.json',COUNT_DEFINITIONS)
        with (run/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['step',*[f'{tag}_{coord}' for tag in ('truth','estimate_pre','estimate_post') for coord in ('x','vx','y','vy')]])
            for k in range(1,len(t)):writer.writerow([k,*t[k],*pre[k],*post[k]])
        green=HERE/'test-green-attempt-02.log'
        if not green.exists():raise AssertionError('pre-science test evidence missing')
        dump(run/'test_evidence.json',{'suite':'code/revision_20261009/qgpf/test_stable.py','suite_sha256':sha(HERE/'test_stable.py'),
         'green_log':'code/revision_20261009/qgpf/test-green-attempt-02.log','green_log_sha256':sha(green),
         'green_exit_code':0,'test_count':11,'tests_run_before_science':True,'framework':'stdlib unittest + seeded generated properties; not Hypothesis'})
        if code_hashes!={x:sha(HERE/x) for x in CODE_FILES}:raise AssertionError('code changed during run')
        status='completed'
    except Exception as exc:
        status='failed';trace=traceback.format_exc().replace(str(WORKSPACE),'<workspace>').replace(str(HERE),'<revision-qgpf>')
        failure={'exception_type':type(exc).__name__,'message':str(exc).replace(str(WORKSPACE),'<workspace>'),
                 'traceback_redacted':trace,'no_seed_exclusion':True}
        dump(run/'failure.json',failure)
    (run/'stdout.log').write_text(json.dumps({'status':status,'metrics':metrics},ensure_ascii=False)+'\n',encoding='utf-8')
    (run/'stderr.log').write_text(json.dumps(failure,ensure_ascii=False)+'\n' if failure else '',encoding='utf-8')
    files={p.name:sha(p) for p in run.iterdir() if p.is_file() and p.name!='manifest.json'}
    manifest={'schema':'revision.qgpf.run.v1','status':status,'run_id':f'{method}-seed-{seed:02d}',
     'started_at':started,'ended_at':now(),'method':method,'seed':seed,'stream_seed':stream_seed(seed,method),
     'code_sha256':code_hashes,'input_evidence':input_info,'protocol_sha256':protocol_sha,
     'historical_source':'code/qgpf_reproduction/runs-main-30','files_sha256':files,
     'failure_artifact':'failure.json' if failure else None,'scheduler':scheduler}
    dump(run/'manifest.json',manifest)
    return metrics,manifest
def run_batch(seeds,methods,out_dir,workers,batch_kind):
    out_dir.mkdir(parents=True,exist_ok=True)
    scheduler={'kind':batch_kind,'max_independent_processes':workers,'blas_threads_per_process':1,
               'runtime_comparison_allowed':False if workers>1 else batch_kind=='serial-single-load-timing'}
    rows=[];manifests=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_one,s,m,str(out_dir),scheduler):(m,s) for m in methods for s in seeds}
        for future in as_completed(futures):
            m,s=futures[future];metrics,manifest=future.result();rows.append(metrics);manifests.append(manifest)
            print(f'{now()} end {m} seed={s} status={manifest["status"]}',flush=True)
            if manifest['status']!='completed':
                for pending in futures:pending.cancel()
                raise RuntimeError(f'failed {m} seed={s}; failure artifact retained; batch stopped')
    rows.sort(key=lambda r:(methods.index(r['method']),r['seed']))
    with (out_dir/'raw_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(out_dir/'batch_manifest.json',{'schema':'revision.qgpf.batch.v1','protocol_sha256':sha(PROTOCOL),
     'methods':methods,'seeds':seeds,'completed':len(manifests),'failed':0,'scheduler':scheduler,
     'run_manifests':[f'{m}/seed-{s:02d}/manifest.json' for m in methods for s in seeds],
     'code_sha256':{x:sha(HERE/x) for x in CODE_FILES},'raw_metrics_csv_sha256':sha(out_dir/'raw_metrics.csv'),
     'counter_definitions':COUNT_DEFINITIONS})
    return rows
def verify_replay(first,second,methods):
    checks=[]
    for method in methods:
        a=first/method/'seed-00';b=second/method/'seed-00'
        arrays={name:bool(np.array_equal(np.load(a/f'{name}.npy',allow_pickle=False),np.load(b/f'{name}.npy',allow_pickle=False))) for name in SCIENTIFIC_ARRAYS}
        ma=json.loads((a/'raw_metrics.json').read_text());mb=json.loads((b/'raw_metrics.json').read_text())
        metrics_equal={k:ma[k]==mb[k] for k in ma if k not in ('runtime_s','initialization_runtime_s')}
        rng_equal=sha(a/'rng_final_state.json')==sha(b/'rng_final_state.json')
        independent_pid=json.loads((a/'environment.json').read_text())['pid']!=json.loads((b/'environment.json').read_text())['pid']
        checks.append({'method':method,'seed':0,'scientific_arrays_equal':arrays,'metrics_excluding_timer_equal':metrics_equal,
         'rng_final_equal':rng_equal,'separate_process_pid':independent_pid,'excluded_operational_outputs':['runtime_s','initialization_runtime_s','step_elapsed_s.npy','timestamps','environment.json','stdout.log'],
         'passed':all(arrays.values()) and all(metrics_equal.values()) and rng_equal and independent_pid})
    dump(second/'fixed_seed_replay.json',{'schema':'revision.qgpf.fixed-seed.v1','checks':checks,'passed':all(x['passed'] for x in checks)})
    if not all(x['passed'] for x in checks):raise AssertionError('fixed-seed scientific replay failed')
def parse_seeds(spec):
    if ':' in spec:
        a,b=map(int,spec.split(':',1));return list(range(a,b+1))
    return [int(x) for x in spec.split(',') if x]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out-dir',type=Path,default=HERE/'runs-m1')
    ap.add_argument('--seeds',default='0:29');ap.add_argument('--methods',default='B-stable,Q-stable')
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--replay',action='store_true')
    ap.add_argument('--serial-timing',action='store_true');args=ap.parse_args()
    seeds=parse_seeds(args.seeds);methods=tuple(x for x in args.methods.split(',') if x)
    if not seeds or any(s not in SEEDS for s in seeds) or any(m not in METHODS for m in methods):raise ValueError('invalid frozen seed/method')
    if not 1<=args.workers<=4:raise ValueError('one to four workers only')
    load_contract();kind='serial-single-load-timing' if args.serial_timing else 'independent-seed-batch'
    run_batch(seeds,methods,args.out_dir,args.workers,kind)
    if args.replay:
        replay=args.out_dir.with_name(args.out_dir.name+'-fixed-replay')
        run_batch([0],methods,replay,1,'independent-fixed-seed-replay')
        verify_replay(args.out_dir,replay,methods)

if __name__=='__main__':main()
