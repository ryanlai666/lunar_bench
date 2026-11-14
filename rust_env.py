"""ctypes interface to RustLander-v0 (a custom surrogate, NOT Box2D LunarLander)."""
from __future__ import annotations
import ctypes as ct
from pathlib import Path
import sys
import numpy as np

ENVIRONMENT_ID = 'RustLander-v0'
OBS_SIZE = 8
N_ACTIONS = 4
OUT_SIZE = 12

def library_path():
    name = ('rust_lander.dll' if sys.platform == 'win32' else
            'librust_lander.dylib' if sys.platform == 'darwin' else 'librust_lander.so')
    return Path(__file__).resolve().parent / 'native' / 'target' / 'release' / name

def _integer(value, low, high, label):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or not low <= value <= high:
        raise ValueError(f'{label} must be an integer in [{low}, {high}]')
    return int(value)

class RustBatchEnv:
    """Native batch, no automatic reset; one FFI call per batch step."""
    def __init__(self, count=8, horizon=1000):
        count=_integer(count,1,65536,'count')
        horizon=_integer(horizon,1,2**32-1,'horizon')
        self.lib=ct.CDLL(str(library_path()))
        fp=ct.POINTER(ct.c_float)
        for name,args,result in (
            ('rl_version',[],ct.c_uint32),
            ('rl_create',[ct.c_uint32,ct.c_uint32],ct.c_void_p),
            ('rl_destroy',[ct.c_void_p],None),
            ('rl_reset',[ct.c_void_p,ct.c_uint32,ct.c_uint64,fp],ct.c_int),
            ('rl_step',[ct.c_void_p,ct.POINTER(ct.c_uint32),fp],ct.c_int)):
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=result
        if self.lib.rl_version()!=1: raise RuntimeError('RustLander ABI mismatch')
        self.count,self.horizon=count,horizon
        self.output=np.zeros((count,OUT_SIZE),dtype=np.float32)
        self.steps=np.zeros(count,dtype=np.int64)
        self.ready=np.zeros(count,dtype=bool)
        self.handle=self.lib.rl_create(count,horizon)
        if not self.handle: raise RuntimeError('RustLander creation failed')
        self.rng=np.random.default_rng()

    def _open(self):
        if not self.handle: raise RuntimeError('Environment is closed')

    def reset_at(self,index,seed):
        self._open()
        index=_integer(index,0,self.count-1,'index');seed=_integer(seed,0,2**64-1,'seed')
        code=self.lib.rl_reset(self.handle,index,seed,self.output[index].ctypes.data_as(ct.POINTER(ct.c_float)))
        if code: raise RuntimeError(f'Native reset failed: {code}')
        self.steps[index]=0;self.ready[index]=True
        return self.output[index,:OBS_SIZE].copy()

    def step(self,actions):
        self._open()
        if not self.ready.all(): raise RuntimeError('Reset new or ended environments before stepping')
        a=np.asarray(actions)
        if a.shape!=(self.count,) or not np.issubdtype(a.dtype,np.integer) or (a<0).any() or (a>=N_ACTIONS).any():
            raise ValueError('One integer action 0..3 required per environment')
        a=np.ascontiguousarray(a,dtype=np.uint32)
        code=self.lib.rl_step(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_uint32)),self.output.ctypes.data_as(ct.POINTER(ct.c_float)))
        if code: raise RuntimeError(f'Native step failed: {code}')
        self.steps+=1
        term=self.output[:,9].astype(bool);trunc=self.output[:,10].astype(bool)
        self.ready=~(term|trunc)
        return self.output[:,:OBS_SIZE].copy(),self.output[:,8].copy(),term,trunc

    def info(self,index):
        index=_integer(index,0,self.count-1,'index')
        success=bool(self.output[index,11])
        return dict(success=success,landed=success,crashed=bool(self.output[index,9]) and not success,
                    steps=int(self.steps[index]),environment=ENVIRONMENT_ID)

    def close(self):
        if getattr(self,'handle',None):
            self.lib.rl_destroy(self.handle);self.handle=None
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
    def __del__(self): self.close()

class RustEnv(RustBatchEnv):
    def __init__(self,horizon=1000): super().__init__(1,horizon)
    def reset(self,*,seed=None,options=None):
        if options: raise ValueError('Reset options are not supported')
        if seed is None: seed=int(self.rng.integers(0,2**63))
        return self.reset_at(0,seed),self.info(0)
    def step(self,action):
        a=_integer(action,0,3,'action')
        obs,r,t,u=super().step([a])
        return obs[0],float(r[0]),bool(t[0]),bool(u[0]),self.info(0)

def heuristic(obs):
    """PD controller adapted to this surrogate's units; reference, not learned."""
    x,y,vx,vy,angle,omega,*_=obs
    target_angle=np.clip(1.3*x+1.7*vx,-0.5,0.5)
    turn=3.0*(target_angle-angle)-1.5*omega
    target_vy=-min(0.6,max(0.08,0.9*y))
    if vy < target_vy and abs(angle)<0.7:
        return 2
    if turn>0.12: return 1
    if turn < -0.12: return 3
    return 0
