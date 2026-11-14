"""Nonlearning references on the same held-out episode seeds."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,json,time
from pathlib import Path
import numpy as np
from .run import make_env

def main():
    rows=[]
    for backend in ['gym','rust']:
        for policy in ['random','heuristic']:
            env=make_env(backend)
            if backend=='gym':
                from gymnasium.envs.box2d.lunar_lander import heuristic
                control=lambda obs:heuristic(env.unwrapped,obs)
            else:
                from .rust_env import heuristic
                control=heuristic
            for seed in range(2000000000,2000000100):
                rng=np.random.default_rng(seed+3000000000)
                obs,_=env.reset(seed=seed);score=0.;start=time.perf_counter()
                for length in range(1,1001):
                    action=int(rng.integers(4)) if policy=='random' else int(control(obs))
                    obs,r,t,u,info=env.step(action);score+=float(r)
                    if t or u:break
                rows.append(dict(backend=backend,policy=policy,seed=seed,return_=score,length=length,
                                 success=bool(info.get('success',t and r>=99)),seconds=time.perf_counter()-start))
            env.close()
            print(backend,policy,float(np.mean([r['return_'] for r in rows if r['backend']==backend and r['policy']==policy])),flush=True)
    out=Path('lunar_bench/results/baselines');out.mkdir(exist_ok=True)
    with (out/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for backend in ['gym','rust']:
        summary[backend]={}
        for policy in ['random','heuristic']:
            subset=[r for r in rows if r['backend']==backend and r['policy']==policy]
            summary[backend][policy]={'mean_return':float(np.mean([r['return_'] for r in subset])),
                                      'success_rate':float(np.mean([r['success'] for r in subset])),
                                      'episodes':len(subset)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
