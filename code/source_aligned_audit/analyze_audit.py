"""Recompute frozen seed-level summaries and paired uncertainty from raw arrays."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from run_audit import save_json

BASE=Path(__file__).resolve().parent
plan=json.loads((BASE/'frozen_plan.json').read_text(encoding='utf-8'))
root=BASE/'runs-main-30'
records=[]
for seed in range(30):
    out=root/f'seed-{seed:02d}'
    truth=np.load(out/'truth.npy')
    seed_summary=json.loads((out/'seed_summary.json').read_text(encoding='utf-8'))
    for method in plan['methods']:
        estimate=np.load(out/f'{method}_estimates.npy')
        recomputed=float(np.sqrt(np.mean(np.sum((estimate-truth)**2,axis=1))))
        summary=next(r for r in seed_summary if r['method']==method)
        np.testing.assert_allclose(recomputed,summary['rmse'],rtol=0,atol=1e-14)
        records.append(summary)
with (root/'seed_metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
summary={}
for method in plan['methods']:
    selected=[r for r in records if r['method']==method]
    numbers=['rmse','mae_euclidean','posterior_mean_rmse','mean_distinct_count','runtime_s',
             'fitness_evaluations','likelihood_evaluations','candidate_distance_evaluations',
             'binary_dimensions_processed','mean_sampling_ess','mean_logfitness_gap']
    summary[method]={}
    for k in numbers:
        vals=[r[k] for r in selected]
        if vals[0] is None:
            summary[method][k]=None
        else:
            summary[method][k]={'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1)),
                              'min':float(min(vals)),'max':float(max(vals))}
bootstrap_indices=np.random.default_rng(9301).integers(0,30,size=(20000,30))
signs=np.random.default_rng(9302).choice(np.array([-1.,1.]),size=(50000,30))
source=np.array([r['rmse'] for r in records if r['method']=='source_ecqpf_operator'])
comparisons=[]
for method in ['ellipse_exact_fitness','ellipse_systematic_fitness']:
    baseline=np.array([r['rmse'] for r in records if r['method']==method])
    difference=source-baseline
    ci=np.quantile(difference[bootstrap_indices].mean(axis=1),[.025,.975])
    p=float((np.count_nonzero(np.abs((signs*difference).mean(axis=1))>=abs(difference.mean()))+1)/(len(signs)+1))
    comparisons.append({'comparison':f'source_ecqpf_operator minus {method}',
        'mean_paired_rmse_difference':float(difference.mean()),'bootstrap95_ci':ci.tolist(),
        'paired_dz':float(difference.mean()/difference.std(ddof=1)),'permutation_p_uncorrected':p,
        'bootstrap_samples':20000,'permutation_samples':50000,
        'paired_rmse_differences':difference.tolist()})
ordering=np.argsort([r['permutation_p_uncorrected'] for r in comparisons])
prev=0.
for rank,index in enumerate(ordering):
    value=max(prev,(len(comparisons)-rank)*comparisons[index]['permutation_p_uncorrected'])
    prev=value
    comparisons[index]['permutation_p_holm']=min(1.,value)
result={'schema':'source_aligned_module_analysis.v1','independent_seed_datasets':30,
        'one_step_updates_per_dataset':20,'successful_updates':600,'failed_seed_datasets':0,
        'plan_sha256':hashlib.sha256((BASE/'frozen_plan.json').read_bytes()).hexdigest(),
        'interpretation_scope':plan['scope'],'rmse_recomputed_from_raw_estimates_and_truth':True,
        'summary':summary,'primary_paired_comparisons':comparisons,
        'source_probability_convention_ambiguity':'Literal Eq12 probability-zero orientation; no sensitivity variant was run.',
        'original_algorithm_efficacy_conclusion':'NOT_ASSESSED','equivalence_conclusion':'NOT_ASSESSED'}
save_json(root/'analysis.json',result)
print(json.dumps({'summary_rmse':{m:r['rmse'] for m,r in summary.items()},'paired':comparisons},indent=2))
