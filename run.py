"""Actual local equal-transition PPO/A2C/DoubleDQN training and evaluation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .agents import PPO,A2C,DoubleDQN,gae


def make_env(backend):
    if backend=='gym':
        import gymnasium as gym
        return gym.make('LunarLander-v3',max_episode_steps=1000)
    from .rust_env import RustEnv
    return RustEnv(horizon=1000)


def evaluate(model,method,backend,episodes,start_seed):
    env=make_env(backend)
    rows=[]
    began=time.perf_counter()
    try:
        for seed in range(start_seed,start_seed+episodes):
            obs,_=env.reset(seed=seed)
            score=0.
            for length in range(1,1001):
                action=(int(model.act(obs[None,:],epsilon=0)[0]) if method=='ddqn'
                        else int(model.act(obs[None,:],deterministic=True)[0][0]))
                obs,r,term,trunc,info=env.step(action)
                score+=float(r)
                if term or trunc: break
            rows.append(dict(seed=seed,return_=score,length=length,
                             success=bool(info.get('success',term and r>=99)),
                             terminated=bool(term),truncated=bool(trunc)))
    finally: env.close()
    return dict(mean_return=float(np.mean([r['return_'] for r in rows])),
                success_rate=float(np.mean([r['success'] for r in rows])),
                seconds=time.perf_counter()-began,episodes=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['ppo','a2c','ddqn'],required=True)
    p.add_argument('--backend',choices=['gym','rust'],default='gym')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=300000)
    p.add_argument('--envs',type=int,default=8)
    p.add_argument('--eval-episodes',type=int,default=100)
    p.add_argument('--report-every',type=int,default=50000)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.steps,a.envs,a.eval_episodes,a.report_every)<=0 or a.steps%a.envs:
        p.error('Positive counts and steps divisible by envs required')
    if a.output.exists(): p.error('Use a fresh output directory')
    a.output.mkdir(parents=True)
    model=({'ppo':PPO,'a2c':A2C,'ddqn':DoubleDQN}[a.method])(seed=a.seed,lr=3e-4)
    cfg={**vars(a),'output':str(a.output),'environment':'LunarLander-v3' if a.backend=='gym' else 'RustLander-v0',
         'python':platform.python_version(),'numpy':np.__version__,'threads_per_process':1,
         'training_seed_range':'[0,1000000000)','validation_seed_start':1000000000,
         'test_seed_start':2000000000,'checkpoint_selection':'final fixed-budget checkpoint',
         'reward_scale_for_optimization':0.01 if a.method!='ddqn' else 1.0,
         'network':'two ReLU hidden layers, 64 units each' if a.method=='ddqn' else 'single tanh hidden layer, 64 units',
         'model_config':model.config,
         'algorithm_source_sha256':hashlib.sha256(Path(__file__).with_name('agents.py').read_bytes()).hexdigest()}
    if a.backend=='gym':
        import gymnasium,Box2D
        cfg.update(gymnasium=gymnasium.__version__,box2d=Box2D.__version__)
    else:
        binary=Path(__file__).parent/'native/target/release/rust_lander.dll'
        cfg['simulator_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    (a.output/'config.json').write_text(json.dumps(cfg,indent=2))
    rng=np.random.default_rng(a.seed+100)
    native_batch=None
    if a.backend=='rust':
        from .rust_env import RustBatchEnv
        native_batch=RustBatchEnv(a.envs,horizon=1000)
        envs=[]
        obs=np.stack([native_batch.reset_at(i,int(rng.integers(1000000000))) for i in range(a.envs)])
    else:
        envs=[make_env(a.backend) for _ in range(a.envs)]
        obs=np.stack([e.reset(seed=int(rng.integers(1000000000)))[0] for e in envs])
    total=0;seconds=0.;cpu_seconds=0.;updates=0;next_report=a.report_every
    episode_scores=np.zeros(a.envs);episode_lengths=np.zeros(a.envs,dtype=int)
    wall_start=time.perf_counter()
    log=(a.output/'training.jsonl').open('w')
    progress=(a.output/'progress.jsonl').open('w')
    def report():
        result=evaluate(model,a.method,a.backend,10,1000000000)
        row=dict(steps=total,training_seconds=seconds,training_cpu_seconds=cpu_seconds,
                 transitions_per_second=total/seconds if seconds else 0.,validation=result)
        progress.write(json.dumps(row)+'\n');progress.flush()
        print(f'{a.backend} {a.method} seed={a.seed} steps={total} return={result["mean_return"]:.1f} success={result["success_rate"]:.0%} steps/s={row["transitions_per_second"]:.0f}',flush=True)
    try:
        report()
        while total<a.steps:
            began=time.perf_counter();cpu_began=time.process_time()
            rollout=[];completed=[]
            length=min(128 if a.method=='ppo' else 32 if a.method=='a2c' else 32,(a.steps-total)//a.envs)
            metrics={}
            for _ in range(length):
                if a.method=='ddqn':
                    epsilon=max(.05,1.-.95*total/(a.steps*.6))
                    actions=model.act(obs,epsilon=epsilon)
                else: actions,logs,values=model.act(obs)
                if native_batch is not None:
                    next_obs,rewards,terms,truncs=native_batch.step(actions)
                    ends=terms|truncs
                    infos=[native_batch.info(i) for i in range(a.envs)]
                else:
                    transitions=[e.step(int(act)) for e,act in zip(envs,actions)]
                    next_obs=np.stack([t[0] for t in transitions])
                    rewards=np.asarray([t[1] for t in transitions])
                    terms=np.asarray([t[2] for t in transitions],dtype=bool)
                    ends=terms|np.asarray([t[3] for t in transitions],dtype=bool)
                    infos=[t[4] for t in transitions]
                episode_scores+=rewards;episode_lengths+=1
                for i in np.flatnonzero(ends):
                    completed.append(dict(return_=float(episode_scores[i]),length=int(episode_lengths[i]),
                                          success=bool(infos[i].get('success',terms[i] and rewards[i]>=99))))
                    episode_scores[i]=0.;episode_lengths[i]=0
                if a.method=='ddqn':
                    model.observe(obs,actions,rewards,next_obs,terms)
                    if total>=5000:
                        for _ in range(max(1,a.envs//4)): metrics=model.update(batch_size=64);updates+=1
                else:
                    next_values=model.forward(next_obs)[3]
                    rollout.append((obs.copy(),actions,logs,values,rewards*.01,next_values,terms,ends))
                for i in np.flatnonzero(ends):
                    seed=int(rng.integers(1000000000))
                    next_obs[i]=(native_batch.reset_at(int(i),seed) if native_batch is not None
                                 else envs[i].reset(seed=seed)[0])
                obs=next_obs;total+=a.envs
            if a.method!='ddqn':
                states,acts,logs,vals,rews,nvals,terms,ends=map(np.stack,zip(*rollout))
                adv,ret=gae(rews,vals,nvals,terms,ends)
                metrics=model.update(states.reshape(-1,8),acts.ravel(),logs.ravel(),adv.ravel(),ret.ravel())
                updates+=1
            seconds+=time.perf_counter()-began;cpu_seconds+=time.process_time()-cpu_began
            log.write(json.dumps(dict(steps=total,training_seconds=seconds,updates=updates,metrics=metrics,episodes=completed))+'\n');log.flush()
            if total>=next_report or total==a.steps:
                report();next_report+=a.report_every
        model.save(a.output/'final.npz',cfg)
        result=evaluate(model,a.method,a.backend,a.eval_episodes,2000000000)
        summary=dict(config=cfg,trained_steps=total,training_seconds=seconds,training_cpu_seconds=cpu_seconds,
                     transitions_per_second=total/seconds,wall_seconds=time.perf_counter()-wall_start,
                     optimizer_updates=model.updates,update_calls=updates,test=result)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        with (a.output/'test_episodes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result['episodes'][0]));w.writeheader();w.writerows(result['episodes'])
        print(f'DONE {a.backend} {a.method} seed={a.seed}: test={result["mean_return"]:.1f}',flush=True)
    finally:
        log.close();progress.close()
        for e in envs:e.close()
        if native_batch is not None:native_batch.close()

if __name__=='__main__':main()
