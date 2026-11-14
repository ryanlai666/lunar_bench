"""Uncontended simulator throughput: run AFTER training workers finish.

Gymnasium Box2D and RustLander-v0 do not implement equivalent dynamics; these are
workload measurements, not a language-only or faithful-port speed comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import gymnasium as gym
from .rust_env import RustEnv,RustBatchEnv,library_path


def scalar(factory,steps,seed):
    actions=np.random.default_rng(seed).integers(0,4,size=steps)
    env=factory();env.reset(seed=seed)
    episodes=0;checksum=0.
    began=time.perf_counter()
    try:
        for action in actions:
            _,r,t,u,_=env.step(int(action));checksum+=float(r)
            if t or u:
                episodes+=1;env.reset(seed=seed+episodes)
    finally:env.close()
    seconds=time.perf_counter()-began
    return dict(steps=steps,seconds=seconds,steps_per_second=steps/seconds,episodes=episodes,return_checksum=checksum)


def batch(steps,seed,count):
    rounds=(steps+count-1)//count
    actions=np.random.default_rng(seed).integers(0,4,size=(rounds,count),dtype=np.uint32)
    env=RustBatchEnv(count)
    for i in range(count):env.reset_at(i,seed+i)
    episodes=0;checksum=0.
    began=time.perf_counter()
    try:
        for a in actions:
            _,r,t,u=env.step(a);checksum+=float(r.sum())
            for i in np.flatnonzero(t|u):
                episodes+=1;env.reset_at(int(i),seed+count+episodes)
    finally:env.close()
    seconds=time.perf_counter()-began;steps=rounds*count
    return dict(steps=steps,seconds=seconds,steps_per_second=steps/seconds,episodes=episodes,return_checksum=checksum)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--output',type=Path,default=Path('lunar_bench/results/throughput.json'))
    p.add_argument('--load-note',default='Run after all training workers have exited; OS/background load is uncontrolled.')
    a=p.parse_args()
    if min(a.steps,a.repeats,a.batch_size)<=0:p.error('positive values required')
    if a.output.exists():p.error('Use a fresh output path to preserve previous measurements')
    gym_factory=lambda:gym.make('LunarLander-v3',render_mode=None)
    # Warm up libraries, world construction and Python dispatch, outside timing.
    scalar(gym_factory,1000,999);scalar(RustEnv,1000,999);batch(1024,999,a.batch_size)
    modes=['gymnasium_box2d_scalar','rust_surrogate_scalar','rust_surrogate_batch']
    rows=[]
    for repeat in range(a.repeats):
        seed=100000+repeat*100000
        # Rotate order to reduce systematic warmup/order bias.
        for mode in modes[repeat%3:]+modes[:repeat%3]:
            r=(scalar(gym_factory,a.steps,seed) if mode==modes[0] else
               scalar(RustEnv,a.steps,seed) if mode==modes[1] else
               batch(a.steps,seed,a.batch_size))
            r.update(mode=mode,repeat=repeat,seed=seed);rows.append(r);print(json.dumps(r),flush=True)
    summary={}
    for mode in modes:
        rates=np.asarray([r['steps_per_second'] for r in rows if r['mode']==mode])
        summary[mode]=dict(mean_steps_per_second=float(rates.mean()),median_steps_per_second=float(np.median(rates)),std_steps_per_second=float(rates.std(ddof=1)) if len(rates)>1 else 0.,min_steps_per_second=float(rates.min()),max_steps_per_second=float(rates.max()))
    report=dict(
        note='Non-equivalent workloads: Gymnasium LunarLander-v3 uses Box2D collision/leg dynamics; RustLander-v0 is a custom flat-ground point-body surrogate. No language-only or faithful-port speedup claim.',
        timing_scope='Step calls, reward checksum, explicit episode resets, and close; excludes action generation, initial creation/reset, and warmup. Rust batch includes ended-index scan and per-environment resets. Rendering disabled.',
        load_note=a.load_note,batch_size=a.batch_size,records=rows,summary=summary,
        machine=dict(platform=platform.platform(),machine=platform.machine(),processor=os.environ.get('PROCESSOR_IDENTIFIER',platform.processor()),logical_cpu_count=os.cpu_count(),python=sys.version,python_executable=sys.executable,numpy=np.__version__,gymnasium=gym.__version__,thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}),
        rust_library_sha256=hashlib.sha256(library_path().read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
