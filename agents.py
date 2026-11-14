"""NumPy PPO, synchronous A2C and Double DQN for local LunarLander experiments.

Time-limit transitions bootstrap their final observation value; episode boundaries
stop GAE recursion. True deaths do not bootstrap. No auto-reset observation leaks.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def gae(rewards, values, next_values, terminated, ended, gamma=0.99, lam=0.95):
    rewards, values, next_values = map(np.asarray, (rewards, values, next_values))
    advantages = np.zeros_like(rewards, dtype=np.float64)
    carry = np.zeros(rewards.shape[1:])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * lam * (~ended[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


class PPO:
    def __init__(self, obs_size=8, actions=4, hidden=64, seed=0, lr=3e-4,
                 clip=0.2, entropy=0.01, value_coef=0.5):
        self.config = dict(obs_size=obs_size, actions=actions, hidden=hidden, seed=seed,
                           lr=lr, clip=clip, entropy=entropy, value_coef=value_coef)
        self.rng = np.random.default_rng(seed)
        self.lr, self.clip, self.entropy, self.value_coef = lr, clip, entropy, value_coef
        self.params = {
            "w": self.rng.normal(0, np.sqrt(2 / obs_size), (obs_size, hidden)),
            "b": np.zeros(hidden),
            "pi": self.rng.normal(0, 0.01 / np.sqrt(hidden), (hidden, actions)),
            "pb": np.zeros(actions),
            "v": self.rng.normal(0, 1 / np.sqrt(hidden), (hidden,)),
            "vb": np.zeros(1),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.s = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.updates = 0

    def forward(self, obs):
        x = np.atleast_2d(obs).astype(np.float64)
        p = self.params
        h = np.tanh(x @ p["w"] + p["b"])
        logits = h @ p["pi"] + p["pb"]
        logits -= logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return h, np.exp(log_probs), log_probs, h @ p["v"] + p["vb"][0]

    def act(self, obs, deterministic=False):
        _, probs, logs, values = self.forward(obs)
        if deterministic:
            actions = probs.argmax(axis=1)
        else:
            draws = self.rng.random(len(probs))[:, None]
            actions = (draws > probs.cumsum(axis=1)).sum(axis=1).clip(max=probs.shape[1] - 1)
        return actions, logs[np.arange(len(probs)), actions], values

    def loss_grad(self, obs, actions, old_logs, advantages, returns):
        h, probs, logs, values = self.forward(obs)
        n = len(obs)
        rows = np.arange(n)
        ratio = np.exp(np.clip(logs[rows, actions] - old_logs, -60, 60))
        clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip)
        surrogate = np.minimum(ratio * advantages, clipped * advantages)
        active = ((advantages >= 0) & (ratio <= 1 + self.clip)) | (
            (advantages < 0) & (ratio >= 1 - self.clip))
        g_log = -advantages * ratio * active / n
        one_hot = np.eye(probs.shape[1])[actions]
        g_logits = g_log[:, None] * (one_hot - probs)
        entropy = -(probs * logs).sum(axis=1)
        # Gradient of -entropy coefficient * entropy.
        g_logits += self.entropy / n * probs * (logs + entropy[:, None])
        value_error = values - returns
        g_value = self.value_coef * value_error / n
        p = self.params
        g_h = (g_logits @ p["pi"].T + g_value[:, None] * p["v"]) * (1 - h*h)
        grads = {
            "pi": h.T @ g_logits, "pb": g_logits.sum(axis=0),
            "v": h.T @ g_value, "vb": np.array([g_value.sum()]),
            "w": np.asarray(obs).T @ g_h, "b": g_h.sum(axis=0),
        }
        metrics = dict(loss=float(-surrogate.mean() + self.value_coef * 0.5 * (value_error**2).mean()
                                  - self.entropy * entropy.mean()),
                       entropy=float(entropy.mean()),
                       approx_kl=float(((ratio - 1) - (logs[rows, actions] - old_logs)).mean()),
                       clip_fraction=float((np.abs(ratio - 1) > self.clip).mean()))
        return metrics, grads

    def update(self, obs, actions, old_logs, advantages, returns, epochs=4, batch_size=128, target_kl=0.03):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        history = []
        for _ in range(epochs):
            order = self.rng.permutation(len(obs))
            for start in range(0, len(obs), batch_size):
                idx = order[start:start + batch_size]
                metrics, grads = self.loss_grad(obs[idx], actions[idx], old_logs[idx], advantages[idx], returns[idx])
                history.append(metrics)
                if metrics["approx_kl"] > target_kl:
                    return {k: float(np.mean([m[k] for m in history])) for k in metrics}
                norm = np.sqrt(sum(np.square(g).sum() for g in grads.values()))
                self.updates += 1
                for key, grad in grads.items():
                    grad = grad * min(1.0, 0.5 / (norm + 1e-8))
                    self.m[key] = 0.9 * self.m[key] + 0.1 * grad
                    self.s[key] = 0.999 * self.s[key] + 0.001 * grad * grad
                    mh = self.m[key] / (1 - 0.9**self.updates)
                    sh = self.s[key] / (1 - 0.999**self.updates)
                    self.params[key] -= self.lr * mh / (np.sqrt(sh) + 1e-8)
                if not all(np.isfinite(p).all() for p in self.params.values()):
                    raise FloatingPointError("PPO parameters became non-finite")
        return {k: float(np.mean([m[k] for m in history])) for k in history[0]}

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": np.array(json.dumps(self.config)),
                   "metadata": np.array(json.dumps(metadata or {})),
                   "rng": np.array(json.dumps(self.rng.bit_generator.state)),
                   "updates": np.array(self.updates)}
        for prefix, values in (("p", self.params), ("m", self.m), ("s", self.s)):
            payload.update({f"{prefix}_{k}": v for k, v in values.items()})
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(**json.loads(str(data["config"])))
            for prefix, values in (("p", model.params), ("m", model.m), ("s", model.s)):
                for key in values:
                    values[key] = data[f"{prefix}_{key}"].copy()
            model.updates = int(data["updates"])
            model.rng.bit_generator.state = json.loads(str(data["rng"]))
            model.metadata = json.loads(str(data["metadata"]))
        return model

class A2C(PPO):
    """Synchronous actor-critic: one vanilla policy-gradient update per rollout.

    The runner supplies GAE advantages and bootstrapped value targets. No importance
    ratios, clipping, replay, shuffled minibatches or repeated epochs are used.
    """
    def loss_grad(self, obs, actions, old_logs, advantages, returns):
        h, probs, logs, values = self.forward(obs)
        n = len(obs)
        chosen = logs[np.arange(n), actions]
        entropy = -(probs * logs).sum(axis=1)
        one_hot = np.eye(probs.shape[1])[actions]
        g_logits = -advantages[:, None] * (one_hot-probs) / n
        g_logits += self.entropy / n * probs * (logs+entropy[:, None])
        error = values-returns
        g_value = self.value_coef*error/n
        g_h = (g_logits@self.params['pi'].T + g_value[:,None]*self.params['v'])*(1-h*h)
        grads = dict(pi=h.T@g_logits, pb=g_logits.sum(axis=0),
                     v=h.T@g_value, vb=np.array([g_value.sum()]),
                     w=np.asarray(obs).T@g_h, b=g_h.sum(axis=0))
        metrics = dict(loss=float(-(chosen*advantages).mean()
                                + self.value_coef*.5*np.square(error).mean()
                                - self.entropy*entropy.mean()),
                       entropy=float(entropy.mean()), approx_kl=0., clip_fraction=0.)
        return metrics, grads

    def update(self, obs, actions, old_logs, advantages, returns, **_):
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
        metrics, grads = self.loss_grad(obs,actions,old_logs,advantages,returns)
        norm = np.sqrt(sum(np.square(g).sum() for g in grads.values()))
        self.updates += 1
        for key, grad in grads.items():
            grad *= min(1., .5/(norm+1e-8))
            self.m[key] = .9*self.m[key] + .1*grad
            self.s[key] = .999*self.s[key] + .001*grad*grad
            mh = self.m[key]/(1-.9**self.updates)
            sh = self.s[key]/(1-.999**self.updates)
            self.params[key] -= self.lr*mh/(np.sqrt(sh)+1e-8)
        if not all(np.isfinite(p).all() for p in self.params.values()):
            raise FloatingPointError('A2C parameters became non-finite')
        return metrics


class DoubleDQN:
    """Two-ReLU-layer Double DQN with uniform replay, Huber loss and Adam.

    Only true termination masks bootstrap targets; time limits must be stored
    with their final observations and terminated=False. Online action selection
    and delayed target evaluation implement the Double DQN target.
    """
    def __init__(self, obs_size=8, actions=4, hidden=64, seed=0, lr=5e-4,
                 gamma=.99, capacity=50000, target_interval=250):
        self.config = dict(obs_size=obs_size, actions=actions, hidden=hidden,
                           seed=seed, lr=lr, gamma=gamma, capacity=capacity,
                           target_interval=target_interval)
        self.rng=np.random.default_rng(seed)
        self.lr,self.gamma,self.capacity,self.target_interval=lr,gamma,capacity,target_interval
        self.params={
            'w1':self.rng.normal(0,np.sqrt(2/obs_size),(obs_size,hidden)),
            'b1':np.zeros(hidden),
            'w2':self.rng.normal(0,np.sqrt(2/hidden),(hidden,hidden)),
            'b2':np.zeros(hidden),
            'wq':self.rng.normal(0,.01/np.sqrt(hidden),(hidden,actions)),
            'bq':np.zeros(actions)}
        self.target={k:v.copy() for k,v in self.params.items()}
        self.m={k:np.zeros_like(v) for k,v in self.params.items()}
        self.s={k:np.zeros_like(v) for k,v in self.params.items()}
        self.updates,self.size,self.position=0,0,0
        self.replay_obs=np.zeros((capacity,obs_size),dtype=np.float32)
        self.replay_next=np.zeros_like(self.replay_obs)
        self.replay_actions=np.zeros(capacity,dtype=np.int64)
        self.replay_rewards=np.zeros(capacity,dtype=np.float32)
        self.replay_terminated=np.zeros(capacity,dtype=bool)

    def forward(self,obs,target=False):
        x=np.atleast_2d(obs).astype(np.float64)
        p=self.target if target else self.params
        h1=np.maximum(x@p['w1']+p['b1'],0.)
        h2=np.maximum(h1@p['w2']+p['b2'],0.)
        return h1,h2,h2@p['wq']+p['bq']

    def act(self,obs,epsilon=0.,deterministic=None):
        q=self.forward(obs)[2]
        actions=q.argmax(axis=1)
        if deterministic is True: epsilon=0.
        if epsilon:
            mask=self.rng.random(len(actions))<epsilon
            actions[mask]=self.rng.integers(self.config['actions'],size=int(mask.sum()))
        return actions

    def observe(self,obs,actions,rewards,next_obs,terminated):
        obs,next_obs=np.atleast_2d(obs),np.atleast_2d(next_obs)
        actions,rewards,terminated=map(np.atleast_1d,(actions,rewards,terminated))
        count=len(obs)
        if (obs.shape!=next_obs.shape or obs.shape[1]!=self.config['obs_size']
            or any(len(a)!=count for a in (actions,rewards,terminated))):
            raise ValueError('Inconsistent replay batch shapes')
        if count>self.capacity:
            obs,next_obs=obs[-self.capacity:],next_obs[-self.capacity:]
            actions,rewards,terminated=[a[-self.capacity:] for a in (actions,rewards,terminated)]
            count=self.capacity
        idx=(np.arange(count)+self.position)%self.capacity
        self.replay_obs[idx],self.replay_next[idx]=obs,next_obs
        self.replay_actions[idx],self.replay_rewards[idx]=actions,rewards
        self.replay_terminated[idx]=terminated
        self.position=(self.position+count)%self.capacity
        self.size=min(self.capacity,self.size+count)

    def targets(self,rewards,next_obs,terminated):
        online_actions=self.forward(next_obs)[2].argmax(axis=1)
        target_q=self.forward(next_obs,target=True)[2]
        return np.asarray(rewards)+self.gamma*(~np.asarray(terminated,dtype=bool))*target_q[np.arange(len(target_q)),online_actions]

    def loss_grad(self,obs,actions,targets):
        obs=np.asarray(obs)
        h1,h2,q=self.forward(obs)
        error=q[np.arange(len(q)),actions]-targets
        absolute=np.abs(error)
        loss=np.where(absolute<=1,.5*error*error,absolute-.5).mean()
        gq=np.zeros_like(q)
        gq[np.arange(len(q)),actions]=np.clip(error,-1.,1.)/len(q)
        gh2=(gq@self.params['wq'].T)*(h2>0)
        gh1=(gh2@self.params['w2'].T)*(h1>0)
        grads=dict(wq=h2.T@gq,bq=gq.sum(axis=0),w2=h1.T@gh2,
                   b2=gh2.sum(axis=0),w1=obs.T@gh1,b1=gh1.sum(axis=0))
        return dict(loss=float(loss),mean_abs_td=float(absolute.mean()),mean_q=float(q.mean())),grads

    def update(self,batch_size=64):
        if self.size<batch_size: return dict(loss=0.,mean_abs_td=0.,mean_q=0.,skipped=1)
        idx=self.rng.integers(self.size,size=batch_size)
        target=self.targets(self.replay_rewards[idx],self.replay_next[idx],self.replay_terminated[idx])
        metrics,grads=self.loss_grad(self.replay_obs[idx],self.replay_actions[idx],target)
        norm=np.sqrt(sum(np.square(g).sum() for g in grads.values()))
        self.updates+=1
        for key,grad in grads.items():
            grad*=min(1.,10./(norm+1e-8))
            self.m[key]=.9*self.m[key]+.1*grad
            self.s[key]=.999*self.s[key]+.001*grad*grad
            mh=self.m[key]/(1-.9**self.updates)
            sh=self.s[key]/(1-.999**self.updates)
            self.params[key]-=self.lr*mh/(np.sqrt(sh)+1e-8)
        if not all(np.isfinite(p).all() for p in self.params.values()):
            raise FloatingPointError('DQN parameters became non-finite')
        if self.updates%self.target_interval==0:
            self.target={k:v.copy() for k,v in self.params.items()}
        return metrics

    def save(self,path,metadata=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(config=np.array(json.dumps(self.config)),
                     metadata=np.array(json.dumps(metadata or {})),
                     rng=np.array(json.dumps(self.rng.bit_generator.state)),
                     updates=np.array(self.updates),size=np.array(self.size),position=np.array(self.position))
        for prefix,values in (('p',self.params),('t',self.target),('m',self.m),('s',self.s)):
            payload.update({f'{prefix}_{k}':v for k,v in values.items()})
        # Replay is included so reloaded optimizer state is usable for exact agent continuation.
        for key in ('obs','next','actions','rewards','terminated'):
            payload['replay_'+key]=getattr(self,'replay_'+key)[:self.size]
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('wb') as f: np.savez_compressed(f,**payload)
        temporary.replace(path)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            agent=cls(**json.loads(str(data['config'])))
            for prefix,values in (('p',agent.params),('t',agent.target),('m',agent.m),('s',agent.s)):
                for key in values: values[key]=data[f'{prefix}_{key}'].copy()
            agent.updates,agent.size,agent.position=map(lambda k:int(data[k]),('updates','size','position'))
            agent.rng.bit_generator.state=json.loads(str(data['rng']))
            agent.metadata=json.loads(str(data['metadata']))
            for key in ('obs','next','actions','rewards','terminated'):
                getattr(agent,'replay_'+key)[:agent.size]=data['replay_'+key]
        return agent
