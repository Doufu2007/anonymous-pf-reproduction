"""Recalculate all revision endpoints from saved arrays; family3 waits for C."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,platform,sys
from pathlib import Path
import numpy as np
from scipy import stats
import scipy
from qgpf_stable import COUNTER_NAMES
from run_revision import HERE,REPO,HIST,PROTOCOL,FREEZE,SEEDS,sha,dump,load_contract

PRIMARY='rmse_position'
DESCRIPTIVE=('rmse_position','mae_position','rmse_velocity','mean_ess','mean_ess_over_n',
 'mean_distinct','post_exploratory_rmse_position','post_exploratory_rmse_velocity')
BOOTSTRAP_N=10000
BOOTSTRAP_SEED=20261009

def describe(values):
    a=np.asarray(values,dtype=float);q=np.quantile(a,[.25,.5,.75])
    return {'n':len(a),'mean':float(a.mean()),'sd':float(a.std(ddof=1)),'median':float(q[1]),
     'q1':float(q[0]),'q3':float(q[2]),'iqr':float(q[2]-q[0]),'min':float(a.min()),'max':float(a.max()),
     'skewness_sample':float(stats.skew(a,bias=False)) if a.std()>0 else 0.}
def check_frozen_file(path,freeze):
    assert sha(path)==freeze['inputs'][path.relative_to(REPO.parent).as_posix()]
def load_old(freeze):
    rows=[];hashes={}
    for seed in SEEDS:
        tp=HIST/f'truth_seed{seed}.npy';ep=HIST/f'estimate_qgpf_seed{seed}.npy'
        check_frozen_file(tp,freeze);check_frozen_file(ep,freeze)
        truth=np.load(tp,allow_pickle=False);pre=np.load(ep,allow_pickle=False);e=pre[1:]-truth[1:]
        rows.append({'seed':seed,'method':'Q-old','rmse_position':float(np.sqrt(np.mean(e[:,[0,2]]**2))),
                     'mae_position':float(np.mean(np.abs(e[:,[0,2]]))),'rmse_velocity':float(np.sqrt(np.mean(e[:,[1,3]]**2)))})
        hashes[ep.relative_to(REPO).as_posix()]=sha(ep)
    return rows,hashes
def load_revision(root,methods):
    rows=[];checks=[];hashes={}
    for method in methods:
        for seed in SEEDS:
            run=root/method/f'seed-{seed:02d}'
            manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
            assert manifest['status']=='completed' and manifest['protocol_sha256']==sha(PROTOCOL)
            assert manifest['method']==method and manifest['seed']==seed
            for filename,expected in manifest['files_sha256'].items():
                assert sha(run/filename)==expected,f'run hash mismatch: {method} {seed} {filename}'
            assert {'stdout.log','stderr.log'}.issubset(manifest['files_sha256'])
            truth=np.load(run/'truth.npy',allow_pickle=False);pre=np.load(run/'estimate_pre.npy',allow_pickle=False)
            post=np.load(run/'estimate_post.npy',allow_pickle=False)
            e=pre[1:]-truth[1:];pe=post[1:]-truth[1:]
            metrics=json.loads((run/'raw_metrics.json').read_text(encoding='utf-8'))
            ess=np.load(run/'step_ess.npy',allow_pickle=False);distinct=np.load(run/'step_distinct.npy',allow_pickle=False)
            rebuilt={'rmse_position':float(np.sqrt(np.mean(e[:,[0,2]]**2))),'mae_position':float(np.mean(np.abs(e[:,[0,2]]))),
             'rmse_velocity':float(np.sqrt(np.mean(e[:,[1,3]]**2))),'mean_ess':float(ess.mean()),'mean_ess_over_n':float(ess.mean()/200),
             'mean_distinct':float(distinct.mean()),'post_exploratory_rmse_position':float(np.sqrt(np.mean(pe[:,[0,2]]**2))),
             'post_exploratory_rmse_velocity':float(np.sqrt(np.mean(pe[:,[1,3]]**2)))}
            metric_equal={name:value==metrics[name] for name,value in rebuilt.items()}
            assert all(metric_equal.values())
            particles_pre=np.load(run/'particles_pre.npy',allow_pickle=False);weights_pre=np.load(run/'weights_pre.npy',allow_pickle=False)
            particles_post=np.load(run/'particles_post.npy',allow_pickle=False);weights_post=np.load(run/'weights_post.npy',allow_pickle=False)
            reconstructed_pre=np.asarray([np.sum(w[:,None]*p,axis=0) for w,p in zip(weights_pre,particles_pre)])
            reconstructed_post=np.asarray([np.sum(w[:,None]*p,axis=0) for w,p in zip(weights_post,particles_post)])
            assert np.array_equal(reconstructed_pre,pre[1:]) and np.array_equal(reconstructed_post,post[1:])
            assert np.allclose(weights_pre.sum(axis=1),1.,rtol=0,atol=1e-12) and np.allclose(weights_post.sum(axis=1),1.,rtol=0,atol=1e-12)
            counts=np.load(run/'step_counters.npy',allow_pickle=False).sum(axis=0)
            assert all(int(v)==metrics[name] for name,v in zip(COUNTER_NAMES,counts))
            expected={'likelihood_evals':20000,'candidate_evals':0,'fitness_evals':0,'proposal_chromosome_events':0,
                      'proposal_candidate_attempts':0,'measurement_bit_observations':0,'crossover_operations':0}
            if method!='B-stable':
                expected.update(likelihood_evals=120000,candidate_evals=64000,fitness_evals=64000,
                                proposal_chromosome_events=16000,proposal_candidate_attempts=64000,measurement_bit_observations=5120000)
                assert metrics['accepts']+metrics['rejects']==64000
                assert metrics['rotation_calls']==metrics['probability_entry_updates']
            assert all(metrics[name]==v for name,v in expected.items())
            assert np.isfinite(pre).all() and np.isfinite(post).all()
            checks.append({'method':method,'seed':seed,'all_manifest_file_hashes_match':True,
             'metrics_exact_recalculation':metric_equal,'pre_and_post_exact_from_particles_weights':True,
             'step_counter_sums_match':True,'fixed_operation_counts_match':True,'finite_and_normalized':True})
            rows.append({'seed':seed,'method':method,**rebuilt,**{name:metrics[name] for name in COUNTER_NAMES}})
            hashes[(run/'manifest.json').relative_to(REPO).as_posix()]=sha(run/'manifest.json')
    return rows,checks,hashes
def paired(left,right,label,index):
    d=np.asarray(left)-np.asarray(right);n=len(d);sd=float(d.std(ddof=1));mean=float(d.mean());se=sd/math.sqrt(n)
    critical=float(stats.t.ppf(.975,n-1));test=stats.ttest_1samp(d,0.)
    rng=np.random.default_rng(BOOTSTRAP_SEED+index)
    draws=d[rng.integers(0,n,size=(BOOTSTRAP_N,n))].mean(axis=1)
    return {'comparison':label,'endpoint':PRIMARY,'unit':'30 paired trajectories','n':n,
     'difference_direction':'left minus right; negative means smaller coordinate position RMSE',
     'mean_difference':mean,'sd_difference':sd,'median_difference':float(np.median(d)),
     'paired_t_ci95':[mean-critical*se,mean+critical*se],'seed_bootstrap_percentile_ci95':[float(x) for x in np.quantile(draws,[.025,.975])],
     'bootstrap_draws':BOOTSTRAP_N,'bootstrap_rng_seed':BOOTSTRAP_SEED+index,'paired_dz':mean/sd,
     't_statistic':float(test.statistic),'df':n-1,'p_two_sided_paired_t':float(test.pvalue),
     'left_smaller_rmse':int(np.sum(d<0)),'equal':int(np.sum(d==0)),'left_larger_rmse':int(np.sum(d>0)),
     'seed_differences':d.tolist(),'holm_p_family3':None}
def holm(values):
    order=np.argsort(values);adjusted=np.empty(len(values));running=0.
    for i,index in enumerate(order):
        running=max(running,(len(values)-i)*values[index]);adjusted[index]=min(1.,running)
    return adjusted.tolist()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--include-retain',action='store_true');args=ap.parse_args()
    protocol,freeze=load_contract()
    old,oldhash=load_old(freeze);m1,checks,hashes=load_revision(HERE/'runs-m1',('B-stable','Q-stable'))
    rows=old+m1;allchecks=checks
    if args.include_retain:
        retain,rchecks,rhash=load_revision(HERE/'runs-s1',('Q-stable-retain',));rows+=retain;allchecks+=rchecks;hashes.update(rhash)
        for seed in SEEDS:
            qa=HERE/'runs-m1'/'Q-stable'/f'seed-{seed:02d}'/'initial_particles.npy'
            ra=HERE/'runs-s1'/'Q-stable-retain'/f'seed-{seed:02d}'/'initial_particles.npy'
            assert np.array_equal(np.load(qa,allow_pickle=False),np.load(ra,allow_pickle=False))
    methods=('Q-old','B-stable','Q-stable')+(('Q-stable-retain',) if args.include_retain else ())
    def values(method,key):return [r[key] for r in rows if r['method']==method]
    summaries={}
    for method in methods:
        summaries[method]={key:describe(values(method,key)) for key in DESCRIPTIVE if all(key in r for r in rows if r['method']==method)}
        if method!='Q-old':summaries[method]['operation_counts']={key:describe(values(method,key)) for key in COUNTER_NAMES}
    comparisons=[paired(values('Q-stable',PRIMARY),values('Q-old',PRIMARY),'Q-stable minus Q-old',0),
                 paired(values('Q-stable',PRIMARY),values('B-stable',PRIMARY),'Q-stable minus B-stable',1)]
    if args.include_retain:
        comparisons.append(paired(values('Q-stable-retain',PRIMARY),values('Q-stable',PRIMARY),'Q-stable-retain minus Q-stable',2))
        adjusted=holm([c['p_two_sided_paired_t'] for c in comparisons])
        for comparison,p in zip(comparisons,adjusted):comparison['holm_p_family3']=p
    tag='qgpf-all3' if args.include_retain else 'm1'
    evidence={'schema':'revision.qgpf.analysis.v1','protocol_sha256':sha(PROTOCOL),'historical_freeze_sha256':sha(FREEZE),
     'analysis_code_sha256':sha(__file__),'scientific_run_identity':'code hashes in immutable individual manifests',
     'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__},
     'scope':'Exploratory revision on previously observed archived synthetic inputs; independent reimplementation, not published table reproduction',
     'primary':'coordinate position RMSE = sqrt(mean of squared x/y errors over k=1..100); not Euclidean RMSE',
     'secondary_inference':'Descriptive only; post endpoint does not feed back or use extra likelihood calls',
     'runtime':'Parallel runtime intentionally excluded from inferential/descriptive comparison; single-load serial logs are separate',
     'seed_coverage':{m:[r['seed'] for r in rows if r['method']==m] for m in methods},
     'no_seed_exclusions':True,'recalculation_checks':allchecks,'input_manifest_sha256':{**oldhash,**hashes},
     'summaries':summaries,'comparisons':comparisons,'holm_status':'COMPLETE_FAMILY3' if args.include_retain else 'PENDING_RETENTION_THIRD_COMPARISON; do not interpret unadjusted p as final family result'}
    dump(HERE/f'analysis-{tag}.json',evidence)
    fields=sorted({k for row in rows for k in row})
    with (HERE/f'recalculated-{tag}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# QGPF numerical-revision evidence','','Exploratory revision of an independent reimplementation on archived inputs. The primary is coordinate position RMSE; negative paired differences indicate smaller error. Parallel runtime is excluded from speed claims.','',
           '| Variant | n | Position RMSE mean ± SD | median [Q1, Q3] | Velocity RMSE mean ± SD |','|---|---:|---:|---:|---:|']
    for method in methods:
        s=summaries[method];p=s['rmse_position'];v=s['rmse_velocity']
        lines.append(f"| {method} | {p['n']} | {p['mean']:.6f} ± {p['sd']:.6f} | {p['median']:.6f} [{p['q1']:.6f}, {p['q3']:.6f}] | {v['mean']:.6f} ± {v['sd']:.6f} |")
    lines+=['','| Pair (left minus right) | Δ mean | paired t 95% CI | seed bootstrap 95% CI | dz | left smaller / 30 | Holm family3 p |','|---|---:|---|---|---:|---:|---:|']
    for c in comparisons:
        ci=c['paired_t_ci95'];bi=c['seed_bootstrap_percentile_ci95'];hp=c['holm_p_family3']
        lines.append(f"| {c['comparison']} | {c['mean_difference']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | [{bi[0]:.6f}, {bi[1]:.6f}] | {c['paired_dz']:.6f} | {c['left_smaller_rmse']} | {'pending third comparison' if hp is None else f'{hp:.8g}'} |")
    lines+=['',evidence['holm_status'],'','All saved endpoints were reconstructed exactly from raw arrays. Run file hashes, finite estimates, normalized weights and per-step counter sums passed. Candidate/fitness/proposal scoring labels overlap likelihood scores; rotation/probability-update labels overlap. These are CPU program events, not hardware quantum gates.',
            '','No claim is made that stable ratios recover the source filtering target, unknown implementation choices, or the original published numerical table.']
    (HERE/f'analysis-{tag}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':f'analysis-{tag}.json','summaries':{m:summaries[m]['rmse_position'] for m in methods},'comparisons':[{k:v for k,v in c.items() if k!='seed_differences'} for c in comparisons]},indent=2))

if __name__=='__main__':main()
