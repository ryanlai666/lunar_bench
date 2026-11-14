"""Run independent benchmark processes, with bounded CPU parallelism."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=500000)
    p.add_argument('--jobs',type=int,default=6)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--backends',nargs='+',choices=['gym','rust'],default=['gym','rust'])
    p.add_argument('--methods',nargs='+',choices=['ppo','a2c','ddqn','sac','grpo','gfn'],default=['ppo','a2c','ddqn'])
    p.add_argument('--output',type=Path,default=Path('lunar_bench/results/benchmark_500k'))
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; choose a fresh experiment directory')
    a.output.mkdir(parents=True)
    started=time.perf_counter()
    config=dict(steps_per_run=a.steps,jobs=a.jobs,seeds=a.seeds,backends=a.backends,methods=a.methods,
                logical_cpus=os.cpu_count(),platform=platform.platform(),processor=platform.processor(),
                python=sys.version,threads_per_process=1)
    (a.output/'experiment.json').write_text(json.dumps(config,indent=2))
    tasks=[(backend,method,seed) for backend in a.backends for seed in a.seeds for method in a.methods]
    def run(task):
        backend,method,seed=task
        name=f'{backend}_{method}_seed{seed}'
        command=[sys.executable,'-m','lunar_bench.run','--backend',backend,'--method',method,
                 '--seed',str(seed),'--steps',str(a.steps),'--output',str(a.output/name)]
        env=os.environ.copy()
        for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
        before=time.perf_counter()
        with (a.output/f'{name}.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        return dict(name=name,returncode=result.returncode,wall_seconds=time.perf_counter()-before)
    results=[]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for future in as_completed([pool.submit(run,t) for t in tasks]):
            row=future.result();results.append(row)
            (a.output/'jobs.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(row),flush=True)
    config.update(total_wall_seconds=time.perf_counter()-started,completed=len(results),failures=sum(r['returncode']!=0 for r in results))
    (a.output/'experiment.json').write_text(json.dumps(config,indent=2))
    if config['failures']:raise SystemExit(1)

if __name__=='__main__':main()
