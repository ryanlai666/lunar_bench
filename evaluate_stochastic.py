"""Secondary stochastic-policy evaluation; primary greedy benchmark is preserved."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .agents import PPO,A2C
from .agents_extra import DiscreteSAC,GRPO,GFlowNetTB
from .run import make_env

POLICIES={'ppo':PPO,'a2c':A2C,'sac':DiscreteSAC,'grpo':GRPO,'gfn':GFlowNetTB}   # categorical actors


def evaluate_run(run,episodes=100):
    run=Path(run)
    destination=run/'stochastic_summary.json'
    if destination.exists():
        saved=json.loads(destination.read_text())
        if len(saved['test']['episodes'])!=episodes:
            raise ValueError(f'{run}: existing diagnostic has different episode count')
        return dict(run=run.name,mean_return=saved['test']['mean_return'],cached=True)
    cfg=json.loads((run/'config.json').read_text())
    if cfg['method'] not in POLICIES:raise ValueError('Only categorical actor policies supported')
    model=POLICIES[cfg['method']].load(run/'final.npz')
    env=make_env(cfg['backend']);rows=[];began=time.perf_counter()
    try:
        for seed in range(2000000000,2000000000+episodes):
            model.rng=np.random.default_rng(seed+4000000000)
            obs,_=env.reset(seed=seed);score=0.
            counts=np.zeros(4,dtype=int)
            for length in range(1,1001):
                action=int(model.act(obs[None,:],deterministic=False)[0][0]);counts[action]+=1
                obs,reward,term,trunc,info=env.step(action);score+=float(reward)
                if term or trunc:break
            rows.append(dict(seed=seed,policy_rng_seed=seed+4000000000,return_=score,
                             length=length,success=bool(info.get('success',term and reward>=99)),
                             terminated=bool(term),truncated=bool(trunc),
                             action0=int(counts[0]),action1=int(counts[1]),
                             action2=int(counts[2]),action3=int(counts[3])))
    finally:env.close()
    summary=dict(evaluation_mode='stochastic categorical policy sampling',
                 primary_result_unchanged=True,
                 config=dict(method=cfg['method'],backend=cfg['backend'],seed=cfg['seed'],
                             environment=cfg['environment'],episodes=episodes,
                             environment_seed_start=2000000000,
                             policy_rng='np.random.default_rng(environment_seed + 4000000000), reset every episode',
                             checkpoint_sha256=hashlib.sha256((run/'final.npz').read_bytes()).hexdigest()),
                 test=dict(mean_return=float(np.mean([r['return_'] for r in rows])),
                           success_rate=float(np.mean([r['success'] for r in rows])),
                           seconds=time.perf_counter()-began,episodes=rows))
    with (run/'stochastic_episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temporary=run/'stochastic_summary.tmp'
    temporary.write_text(json.dumps(summary,indent=2));temporary.replace(destination)
    return dict(run=run.name,mean_return=summary['test']['mean_return'],cached=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--backends',nargs='+',choices=['gym','rust'])
    args=p.parse_args()
    if not 1<=args.jobs<=6 or args.episodes<1:p.error('jobs must be 1..6, episodes positive')
    experiment=json.loads((args.directory/'experiment.json').read_text())
    selected=args.backends or experiment['backends']
    methods=[m for m in experiment.get('methods',['ppo','a2c','ddqn']) if m in POLICIES]
    all_runs=[args.directory/f'{b}_{m}_seed{s}' for b in selected
              for m in methods for s in experiment['seeds']]
    incomplete=[str(r) for r in all_runs if not (r/'summary.json').exists()]
    if incomplete:p.error('Wait for primary training/evaluation to complete before this diagnostic: '+', '.join(incomplete))
    targets=[r for r in all_runs if json.loads((r/'config.json').read_text())['method'] in POLICIES]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(evaluate_run,r,args.episodes) for r in targets]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)

if __name__=='__main__':main()
