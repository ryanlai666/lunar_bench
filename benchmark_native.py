"""Measured throughput for custom RustLander-v0, not Box2D speedup."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from .rust_env import RustEnv, RustBatchEnv, library_path

def measure_scalar(steps):
    with RustEnv() as env:
        env.reset(seed=0)
        start=time.perf_counter();episodes=0;checksum=0.
        for i in range(steps):
            _,reward,terminated,truncated,_=env.step(i%4)
            checksum+=reward
            if terminated or truncated:
                episodes+=1;env.reset(seed=episodes)
        seconds=time.perf_counter()-start
    return dict(mode='scalar_ffi',steps=steps,seconds=seconds,steps_per_second=steps/seconds,episodes=episodes,checksum=checksum)

def measure_batch(steps,count):
    rounds=(steps+count-1)//count
    with RustBatchEnv(count) as env:
        for i in range(count):env.reset_at(i,i)
        actions=np.arange(count,dtype=np.uint32)%4
        start=time.perf_counter();episodes=0;checksum=0.
        for j in range(rounds):
            _,r,t,u=env.step(actions)
            checksum+=float(r.sum())
            for i in np.flatnonzero(t|u):
                episodes+=1;env.reset_at(int(i),count+episodes)
        seconds=time.perf_counter()-start
    steps=rounds*count
    return dict(mode=f'batch_{count}_ffi',steps=steps,seconds=seconds,steps_per_second=steps/seconds,episodes=episodes,checksum=checksum)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,default=Path('lunar_bench/results/rust_throughput.json'))
    a=p.parse_args()
    if min(a.steps,a.repeats)<=0:p.error('positive counts required')
    records=[]
    # Warm-up library loading and Python paths outside timed runs.
    measure_scalar(1000);measure_batch(1024,128)
    for repeat in range(a.repeats):
        for run in (measure_scalar(a.steps),measure_batch(a.steps,128)):
            run['repeat']=repeat;records.append(run);print(run,flush=True)
    executable=Path(__file__).parent/'native'/'target'/'release'/('lander-throughput.exe' if sys.platform=='win32' else 'lander-throughput')
    for repeat in range(a.repeats):
        result=json.loads(subprocess.check_output([str(executable),str(a.steps*100)],text=True))
        result.update(mode='native_loop',repeat=repeat);records.append(result);print(result,flush=True)
    summary={mode:dict(mean_steps_per_second=float(np.mean([r['steps_per_second'] for r in records if r['mode']==mode])),std_steps_per_second=float(np.std([r['steps_per_second'] for r in records if r['mode']==mode],ddof=1)) if a.repeats>1 else 0.) for mode in dict.fromkeys(r['mode'] for r in records)}
    report=dict(environment='RustLander-v0',note='Custom surrogate; modes use different action schedules. Native loop excludes FFI/NumPy overhead; scalar and batch include reset handling. Not a Box2D-equivalent acceleration result. Concurrent jobs can affect measurements.',python=platform.python_version(),platform=platform.platform(),simulator_sha256=hashlib.sha256(library_path().read_bytes()).hexdigest(),records=records,summary=summary)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
