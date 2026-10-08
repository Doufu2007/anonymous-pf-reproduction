"""Run the QGPF-like independent benchmark and save raw evidence."""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from qgpf import FilterConfig, ModelConfig, run_filter, simulate_truth

METHODS = ("bootstrap", "qgpf")

def parse_seeds(spec: str) -> list[int]:
    if ":" in spec:
        a,b = (int(x) for x in spec.split(":",1)); return list(range(a,b+1))
    return [int(x) for x in spec.split(",") if x.strip()]

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",default="0:9"); ap.add_argument("--out-dir",type=Path,default=Path("runs/pilot")); args=ap.parse_args()
    seeds=parse_seeds(args.seeds); model=ModelConfig(); cfg=FilterConfig(); args.out_dir.mkdir(parents=True,exist_ok=True)
    code=Path(__file__).with_name("qgpf.py")
    manifest={"schema":"qgpf.independent.reproduction.v1","repro_status":"INDEPENDENT_REIMPLEMENTATION","source":"Wei and Guo (2021), 基于量子遗传粒子滤波的无线传感器网络的目标跟踪算法的研究","python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"code_sha256":sha(code),"model":asdict(model),"filter":asdict(cfg),"seeds":seeds,"methods":METHODS,"assumptions":["The source does not specify measurement-noise variance; set observation_std=1.0.","The reported process-noise diagonal is used as additive standard deviations.","QGA chromosome bits are applied independently to four state coordinates over the explicit bounds.","Elite fraction, generation count, rotation schedule, acceptance rule and N=200 are explicit implementation choices.","Results are a qualitative/independent benchmark and are not a numerical reproduction of Table 1."]}
    (args.out_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    rows=[]
    for seed in seeds:
      truth,obs=simulate_truth(np.random.default_rng(seed),model)
      np.save(args.out_dir/f"truth_seed{seed}.npy",truth); np.save(args.out_dir/f"observations_seed{seed}.npy",obs)
      for mi,method in enumerate(METHODS):
        result=run_filter(obs,truth,np.random.default_rng(seed+1000003*(mi+1)),method,model,cfg)
        np.save(args.out_dir/f"estimate_{method}_seed{seed}.npy",result["estimate"])
        rows.append({"seed":seed,"method":method,**{k:result[k] for k in ("rmse_position","rmse_velocity","rmse_state","mean_ess","mean_distinct","runtime_s","likelihood_evals","candidate_evals","proposal_attempts","accepts")}})
    with (args.out_dir/"raw_metrics.csv").open("w",newline="",encoding="utf-8") as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={}
    for method in METHODS:
      vals=[r for r in rows if r["method"]==method]; summary[method]={"runs":len(vals)}
      for metric in ("rmse_position","rmse_velocity","rmse_state","mean_ess","mean_distinct","runtime_s","likelihood_evals","candidate_evals","proposal_attempts","accepts"):
        arr=np.array([float(v[metric]) for v in vals]); summary[method][metric+"_mean"]=float(arr.mean()); summary[method][metric+"_sd"]=float(arr.std(ddof=1)) if len(arr)>1 else 0.0
    (args.out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
