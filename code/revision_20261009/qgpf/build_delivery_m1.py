from pathlib import Path
import json
from run_revision import HERE,sha,dump
from qgpf_stable import COUNT_DEFINITIONS
analysis=json.loads((HERE/'analysis-m1.json').read_text(encoding='utf-8'))
replay=HERE/'runs-m1-fixed-replay'/'fixed_seed_replay.json'
evidence={'schema':'revision.qgpf.m1-delivery.v1','status':'M1_COMPLETED_AND_VALIDATED',
'completion':{'B-stable':30,'Q-stable':30,'failures':0,'seed_exclusions':0},
'tests':{'suite':'test_stable.py','count':11,'green_exit_code':0,'green_log':'test-green-attempt-02.log','green_log_sha256':sha(HERE/'test-green-attempt-02.log')},
'fixed_seed_replay':{'path':replay.relative_to(HERE).as_posix(),'sha256':sha(replay),'passed':json.loads(replay.read_text(encoding='utf-8'))['passed']},
'files':{name:sha(HERE/name) for name in ['qgpf_stable.py','run_revision.py','test_stable.py','analyze_revision.py','analysis-m1.json','analysis-m1.md','recalculated-m1.csv','runs-m1/batch_manifest.json','runs-m1/raw_metrics.csv','batch-m1-execution.log','public-log-conversion.json','test-red-public.log']},
'descriptive_primary':{method:values['rmse_position'] for method,values in analysis['summaries'].items()},
'comparisons':analysis['comparisons'],'holm_status':analysis['holm_status'],
'counter_definitions':COUNT_DEFINITIONS,
'next_stage':'Intervening EC phase first; Q retention scientific batch waits for parent release after EC validation',
'limitations':['Exploratory archived synthetic inputs','Independent reimplementation, not source numerical recovery','Parallel timers provide no speed comparison','Same-environment repeatability only','Canonical Light stage checkpoint not executed; this delivery is local evidence, not a canonical stage-6 PASS'],
'public_exclusions':['test-red.log','__pycache__/','*.pyc']}
dump(HERE/'delivery-m1.json',evidence)
dump(HERE/'operation-count-definitions.json',COUNT_DEFINITIONS)
print('M1 delivery evidence written')
