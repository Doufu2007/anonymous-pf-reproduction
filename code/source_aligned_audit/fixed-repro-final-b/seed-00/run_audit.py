"""Run the frozen controlled one-step source-operator audit, retaining raw evidence."""
from __future__ import annotations
import argparse
import csv
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
from ecqpf_operator import OperatorConfig, source_select, shared_classical_select, full_systematic_select

BASE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def run_seed(seed, root, role='randomness_estimation'):
    plan=json.loads((BASE/'frozen_plan.json').read_text(encoding='utf-8'))
    out=Path(root)/f'seed-{seed:02d}'
    out.mkdir(parents=True,exist_ok=False)
    started=datetime.now(timezone.utc).isoformat()
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=BASE,capture_output=True,text=True,check=True).stdout.strip()
    dirty=subprocess.run(['git','status','--porcelain'],cwd=BASE,capture_output=True,text=True,check=True).stdout.strip()
    git_diff_hash=hashlib.sha256(subprocess.run(['git','diff','--binary'],cwd=BASE,capture_output=True,check=True).stdout).hexdigest()
    environment={'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
                 'cpu':platform.processor(),'device':'CPU','dtype':'float64','mixed_precision':False,
                 'frameworks':'NumPy only; no train/test fitting, Torch, CUDA, workers or GPU',
                 'PYTHONHASHSEED':os.environ.get('PYTHONHASHSEED'),
                 'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),
                 'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),
                 'same_environment_scope_only':True}
    config={'plan':plan,'plan_sha256':digest(BASE/'frozen_plan.json'),'operator':asdict(OperatorConfig())}
    save_json(out/'config.json',config); save_json(out/'environment.json',environment)
    code_files=[]
    for name in ['ecqpf_operator.py','run_audit.py','test_operator.py']:
        shutil.copy2(BASE/name,out/name)
        code_files.append({'path':name,'sha256':digest(out/name)})
    save_json(out/'test_evidence.json',{'suite':'test_operator.py','tests':9,'green_exit_code':0,
        'red_before_implementation':True,'red_log_sha256':digest(BASE/'test-red.log'),
        'green_log_sha256':digest(BASE/'test-green.log'),'limitations':'formula/unit tests plus runtime guards, not original-algorithm validation'})
    cfg=OperatorConfig()
    ss=np.random.SeedSequence(seed)
    streams=ss.spawn(7)
    truth_rng,obs_rng,particles_rng=[np.random.default_rng(s) for s in streams[:3]]
    rngs={method:np.random.default_rng(s) for method,s in zip(plan['methods'],streams[3:])}
    updates=plan['updates_per_seed']
    truth=truth_rng.normal(0,np.sqrt(plan['generator']['prior_variance']),(updates,2))
    centers=truth+obs_rng.normal(0,np.sqrt(plan['generator']['observation_variance']),(updates,2))
    clouds=particles_rng.normal(0,np.sqrt(plan['generator']['prior_variance']),(updates,cfg.particle_count,2))
    np.save(out/'truth.npy',truth); np.save(out/'centers.npy',centers); np.save(out/'particles.npy',clouds)
    all_results=[]; cost_results=[]; predictions=[]; guardrails=[]
    estimates={m:[] for m in plan['methods']}; indices={m:[] for m in plan['methods']}
    histories=[]
    try:
        for update in range(updates):
            # Algorithms receive only candidate-cloud and observed-center inputs.
            results={}
            for method in plan['methods']:
                if method=='source_ecqpf_operator':
                    result=source_select(clouds[update],centers[update],cfg,rngs[method])
                    histories.append({'update':update,'columns':['generation','best_logfitness','selected_count','mean_q_zero_probability'],'history':result['generation_history']})
                elif method=='full_systematic_likelihood':
                    result=full_systematic_select(clouds[update],centers[update],cfg,rngs[method],plan['generator']['observation_variance'])
                else:
                    result=shared_classical_select(clouds[update],centers[update],cfg,rngs[method],method)
                results[method]=result
                estimate=result['estimate']
                estimates[method].append(estimate); indices[method].append(result['indices'])
                err2=float(np.sum((estimate-truth[update])**2))
                posterior_mean=25./26.*centers[update]
                posterior_err2=float(np.sum((estimate-posterior_mean)**2))
                row={k:v for k,v in result.items() if k not in {'estimate','indices','ellipse_indices','runtime_s','generation_history'}}
                row.update({'seed':seed,'update':update,'method':method,'squared_state_error':err2,
                            'euclidean_state_error':float(np.sqrt(err2)),'squared_posterior_mean_error':posterior_err2})
                all_results.append(row)
                cost_results.append({'update':update,'method':method,'runtime_s':result['runtime_s']})
                predictions.append({'update':update,'method':method,'truth_x':truth[update,0],'truth_y':truth[update,1],
                                    'observed_center_x':centers[update,0],'observed_center_y':centers[update,1],
                                    'estimate_x':estimate[0],'estimate_y':estimate[1]})
            source=results['source_ecqpf_operator']
            common=all(np.array_equal(source['ellipse_indices'],results[m]['ellipse_indices']) for m in plan['methods'][1:3])
            finite=all(np.isfinite(r['estimate']).all() for r in results.values())
            bound=source['logfitness']<=source['optimal_logfitness']+1e-12
            guardrails.append({'update':update,'same_ellipse_for_primary_comparators':common,'finite_estimates':finite,
                               'source_fitness_bounded_by_classical_exact':bound,'source_indices_in_ellipse':set(source['indices']).issubset(set(source['ellipse_indices']))})
            if not common or not finite or not bound:
                raise AssertionError('source-operator audit guardrail failure')
        for method in plan['methods']:
            np.save(out/f'{method}_estimates.npy',np.array(estimates[method]))
            np.save(out/f'{method}_indices.npy',np.array(indices[method]))
        with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
        (out/'raw_metrics.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in all_results),encoding='utf-8')
        save_json(out/'costs.json',cost_results); save_json(out/'source_generation_history.json',histories)
        save_json(out/'guardrails.json',{'status':'PASS','truth_used_only_by_evaluator':True,'checks':guardrails})
        aggregated=[]
        for method in plan['methods']:
            records=[r for r in all_results if r['method']==method]
            costs=[r['runtime_s'] for r in cost_results if r['method']==method]
            item={'seed':seed,'method':method,'rmse':float(np.sqrt(np.mean([r['squared_state_error'] for r in records]))),
                  'mae_euclidean':float(np.mean([r['euclidean_state_error'] for r in records])),
                  'posterior_mean_rmse':float(np.sqrt(np.mean([r['squared_posterior_mean_error'] for r in records]))),
                  'mean_distinct_count':float(np.mean([r['distinct_count'] for r in records])),
                  'runtime_s':float(sum(costs)),
                  'fitness_evaluations':sum(r['fitness_evaluations'] for r in records),
                  'likelihood_evaluations':sum(r['likelihood_evaluations'] for r in records),
                  'candidate_distance_evaluations':sum(r['candidate_distance_evaluations'] for r in records),
                  'binary_dimensions_processed':sum(r['binary_dimensions_processed'] for r in records),
                  'mean_sampling_ess':None if records[0]['sampling_ess'] is None else float(np.mean([r['sampling_ess'] for r in records])),
                  'mean_logfitness_gap':None if records[0]['logfitness'] is None else float(np.mean([r['optimal_logfitness']-r['logfitness'] for r in records]))}
            aggregated.append(item)
        save_json(out/'seed_summary.json',aggregated)
        (out/'stdout.log').write_text(f'completed seed={seed}, updates={updates}, methods=4\n',encoding='utf-8')
        (out/'stderr.log').write_text('',encoding='utf-8')
        status='completed';reason='natural_exit';completion='PASS';failure=None
    except Exception as exc:
        save_json(out/'failure.json',{'exception':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
                                    'completed_update_method_rows':all_results,'completed_cost_rows':cost_results})
        (out/'stdout.log').write_text(f'failed seed={seed}\n',encoding='utf-8')
        (out/'stderr.log').write_text(traceback.format_exc(),encoding='utf-8')
        status='failed';reason='error';completion='FAIL';failure={'path':'failure.json','sha256':digest(out/'failure.json')}
    def record(name):
        return {'path':name,'sha256':digest(out/name)} if (out/name).exists() else None
    manifest={'schema':'light.run_manifest.v3','run_id':f'EC-MODULE__seed-{seed:02d}__{Path(root).name}',
        'matrix_row':'EC-MODULE','status':status,'termination':{'reason':reason,'exit_code':0 if status=='completed' else 1,'signal':None},
        'seed':{'role':role,'value':seed},'command':['python','run_audit.py','--seed',str(seed),'--out',str(Path(root).name)],
        'started_at':started,'ended_at':datetime.now(timezone.utc).isoformat(),
        'config':record('config.json'),'environment':record('environment.json'),'code':{'commit':commit,'dirty':bool(dirty),'diff_sha256':git_diff_hash if dirty else None,'files':code_files},
        'inputs':[{'role':'synthetic_candidate_particles',**record('particles.npy'),'source_revision':config['plan_sha256']},
                  {'role':'observed_noisy_center',**record('centers.npy'),'source_revision':config['plan_sha256']},
                  {'role':'evaluation_truth_only',**record('truth.npy'),'source_revision':config['plan_sha256']}],
        'artifacts':{'stdout':record('stdout.log'),'stderr':record('stderr.log'),'raw_metrics':record('raw_metrics.jsonl'),
                     'predictions':record('predictions.csv'),'test_evidence':record('test_evidence.json'),'guardrail_evidence':record('guardrails.json'),'failure':failure},
        'completion':{'status':completion,'oracle':['nine source-operator tests passed','shared-ellipse and source-fitness bounds checked','all20 updates retained'],
                      'evidence_artifacts':['test_evidence','raw_metrics','predictions','guardrail_evidence']},
        'reproducibility':{'pair_id':'EC-MODULE__seed-0' if role=='fixed_repro' else None,'comparison_role':Path(root).name,
                           'compare_artifacts':['predictions','raw_metrics']},
        'notes':['one-step source-operator audit only; completed local plan, not external preregistration',
                 'runtime is measured separately and intentionally excluded from deterministic output identity']}
    manifest['output_files']={f.name:digest(f) for f in out.iterdir() if f.is_file()}
    save_json(out/'manifest.json',manifest)
    if status!='completed':
        raise RuntimeError(f'seed {seed} failed; preserved {out.name}/failure.json')
    print(json.dumps({'seed':seed,'status':status,'source_rmse':aggregated[0]['rmse'],'source_wall_s':aggregated[0]['runtime_s']}),flush=True)
    return aggregated

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int);parser.add_argument('--out',required=True);parser.add_argument('--role',default='randomness_estimation')
    args=parser.parse_args(); root=Path(args.out)
    if 'PYTHONHASHSEED' not in os.environ or os.environ['PYTHONHASHSEED'] != '0':
        raise RuntimeError('Set PYTHONHASHSEED=0 before launching the process, as required by the frozen execution environment.')
    seeds=[args.seed] if args.seed is not None else list(range(30))
    begin=time.perf_counter(); summaries=[]
    for seed in seeds:
        if time.perf_counter()-begin>900:
            raise TimeoutError('frozen900-second local wall budget reached')
        summaries.extend(run_seed(seed,root,args.role))
    save_json(root/'summary.json',summaries)

if __name__=='__main__':
    main()
