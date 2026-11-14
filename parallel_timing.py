"""Matched small workload: sequential vs six-process actual end-to-end runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=32768)
    p.add_argument('--output',type=Path,default=Path('lunar_bench/results/parallel_timing'))
    a=p.parse_args()
    if a.output.exists():p.error('Choose a fresh output directory')
    a.output.mkdir(parents=True)
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
    tasks=[(m,s) for s in [0,1] for m in ['ppo','a2c','ddqn']]
    times={}
    for jobs in [1,6]:
        def run(task):
            method,seed=task
            out=a.output/f'jobs{jobs}_{method}_{seed}'
            cmd=[sys.executable,'-m','lunar_bench.run','--method',method,'--seed',str(seed),
                 '--steps',str(a.steps),'--eval-episodes','3','--report-every',str(a.steps),'--output',str(out)]
            with out.with_suffix('.log').open('w') as f:
                subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        began=time.perf_counter()
        with ThreadPoolExecutor(max_workers=jobs) as pool:list(pool.map(run,tasks))
        times[jobs]=time.perf_counter()-began
        print(f'jobs={jobs} wall_seconds={times[jobs]:.3f}',flush=True)
    matching=True
    for method,seed in tasks:
        with np.load(a.output/f'jobs1_{method}_{seed}/final.npz') as x,np.load(a.output/f'jobs6_{method}_{seed}/final.npz') as y:
            keys=[k for k in x.files if k.startswith('p_')]
            matching=matching and bool(keys) and all(np.array_equal(x[k],y[k]) for k in keys)
    result=dict(steps_per_run=a.steps,runs_per_mode=len(tasks),serial_seconds=times[1],parallel_seconds=times[6],
                measured_speedup=times[1]/times[6],parameters_identical=matching,
                note='One matched timing pair, CPU NumPy/Gymnasium; includes startup and evaluation; no claim of universal scaling. Sequential measured first, cache/order effects possible.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
