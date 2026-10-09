"""Export transparent publication tables from the completed family3 evidence."""
from pathlib import Path
import csv,json
from run_revision import HERE,sha,dump

def write(name,rows):
    fields=list(rows[0])
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def main():
    source=HERE/'analysis-qgpf-all3.json';a=json.loads(source.read_text(encoding='utf-8'))
    assert a['holm_status']=='COMPLETE_FAMILY3'
    desc=[];count=[];diag=[]
    for method,summary in a['summaries'].items():
        for metric,s in summary.items():
            if metric=='operation_counts':continue
            desc.append({'method':method,'metric':metric,**s,'inference':'primary descriptive' if metric=='rmse_position' else 'exploratory descriptive'})
        if 'operation_counts' not in summary:continue
        for metric,s in summary['operation_counts'].items():
            applicable=metric!='stable_vs_probability_decision_disagreements' or method=='Q-stable'
            row={'method':method,'metric':metric,**s,'applicability':'applicable' if applicable else 'NOT_APPLICABLE; initialized counter zero does not imply agreement'}
            if metric in ('candidate_zero_density_diagnostics','old_zero_density_diagnostics','would_zero_tie_probability_domain',
                          'stable_vs_probability_decision_disagreements','base_zero_density_diagnostics','base_all_zero_density_steps'):
                diag.append(row)
            else:count.append(row)
    paired=[]
    for c in a['comparisons']:
        paired.append({k:v for k,v in c.items() if k not in ('seed_differences','paired_t_ci95','seed_bootstrap_percentile_ci95')})
        paired[-1].update(t_ci95_low=c['paired_t_ci95'][0],t_ci95_high=c['paired_t_ci95'][1],
                         bootstrap_ci95_low=c['seed_bootstrap_percentile_ci95'][0],bootstrap_ci95_high=c['seed_bootstrap_percentile_ci95'][1])
    write('publication-descriptive.csv',desc);write('publication-counts.csv',count)
    write('publication-diagnostics.csv',diag);write('publication-family3.csv',paired)
    dump(HERE/'publication-table-manifest.json',{'schema':'revision.qgpf.publication-tables.v1','analysis_sha256':sha(source),
      'export_code_sha256':sha(__file__),'files_sha256':{x:sha(HERE/x) for x in ('publication-descriptive.csv','publication-counts.csv','publication-diagnostics.csv','publication-family3.csv')},
      'notes':['Primary is coordinate position RMSE; no mixing with Euclidean RMSE',
               'candidate/fitness/proposal scoring labels overlap likelihood; rotation and probability updates overlap',
               'Diagnostics apply to stable trajectories, not historical paired candidate paths',
               'stable_vs_probability_decision_disagreements applies only to ratio Q-stable; other initialized zeros are not agreement evidence',
               'Secondary and post endpoint summaries are exploratory descriptive; no added Holm families',
               'Runtime is omitted; separately reserved single-load samples only']})
    print('Publication table CSVs and manifest exported')
if __name__=='__main__':main()
