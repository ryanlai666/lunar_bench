"""NumPy Discrete SAC, GRPO and GFlowNet trajectory balance (TB) for LunarLander.

Same conventions as agents.py: float64 NumPy, the same hand-written Adam (betas
0.9/0.999), global gradient-norm clipping, policies with one 64-unit tanh layer
and Q networks with two 64-unit ReLU layers. The module is separate from
agents.py so that the algorithm-source hash recorded by the published PPO, A2C
and Double DQN runs still matches. Hyperparameters are fixed defaults chosen
before any run; none were tuned.

Adaptations to LunarLander, where all four discrete actions are valid in every state:

* Discrete SAC. The design this follows used a state-dependent target entropy
  because the number of valid actions varied. Here the action set never changes,
  so the target entropy is the constant 0.5*log(4) ~= 0.69 nats. The paper's
  default of 0.98*log|A| would keep a four-action behaviour policy almost
  uniform (at least 98 % of maximum entropy); 0.5 is an a-priori choice, not a
  tuned value. Critic targets and the actor loss take exact expectations over
  the four actions. Replay is uniform, with Double DQN's capacity, batch size,
  warm-up and update ratio; rewards are unscaled, as for Double DQN.

* GRPO. A group is `group_size` episodes started from the same environment seed
  (the shared root). Every action of episode i gets the advantage
  (R_i - mean R) / std R, with the population std over the group. Groups with an
  episode cut off by the transition budget, or with zero return variance, are
  skipped; their transitions still count toward the budget. The loss is the
  clipped per-action ratio objective plus kl_coef * KL(pi || pi_ref), where
  pi_ref is the FIXED initial policy; there is no critic. With four actions the
  KL is computed exactly instead of with a sampled estimator. Because the
  initial policy is near uniform, the KL term acts much like an entropy bonus.
  Epochs, minibatches, gradient clipping and the approximate-KL early stop
  mirror PPO in agents.py. Every action has equal weight (no per-episode length
  normalisation).

* GFlowNet-TB. The loss is (log Z + sum_t log P_F(a_t|s_t) - log R(tau))^2 with
  a trainable scalar log Z. Each trajectory is treated as a unique path in a
  tree, so log P_B = 0. Returns can be negative, so they are mapped to
      log R = beta * clip((return - r_min) / (r_max - r_min), 0, 1)
  with r_min = -400 and r_max = 300 on both simulators, and
  beta = 1000*log(4) ~= 1386, the largest entropy of a 1000-step action
  sequence. With P_B = 1 the TB optimum is trajectory-level maximum-entropy RL at
  temperature 1/beta. A log-reward range much smaller than the episode's path
  entropy (the 1e-6 + u^4 mapping spans only ~14 nats) would make the optimal
  sampler almost uniformly random over episodes of hundreds of steps. A
  GFlowNet samples trajectories in proportion to R; it does not maximise
  return. log Z is set to its least-squares optimum on the first batch, then
  trained with its own learning rate. Replay keeps the 32 best trajectories by
  return plus the 32 latest ones. After each 1024-transition rollout the agent
  takes 32 TB steps on 16-trajectory minibatches, half from each pool: the same
  number of optimiser steps per rollout as PPO's 4 epochs x 8 minibatches.
  Caveats: the start state differs per episode and Box2D engine dispersion is
  random, so TB's single-root, deterministic-transition assumptions hold only
  approximately with one scalar log Z. With episodes of up to 1000 steps, the
  summed log-probability also makes the TB residual high-variance.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .agents import DoubleDQN


def _policy_init(rng, obs_size, hidden, actions, prefix=''):
    """Same initialisation as the PPO/A2C actor in agents.py."""
    return {prefix+'w': rng.normal(0, np.sqrt(2/obs_size), (obs_size, hidden)),
            prefix+'b': np.zeros(hidden),
            prefix+'pi': rng.normal(0, .01/np.sqrt(hidden), (hidden, actions)),
            prefix+'pb': np.zeros(actions)}


def _policy_forward(p, obs, prefix=''):
    x = np.atleast_2d(obs).astype(np.float64)
    h = np.tanh(x@p[prefix+'w'] + p[prefix+'b'])
    logits = h@p[prefix+'pi'] + p[prefix+'pb']
    logits = logits - logits.max(axis=1, keepdims=True)
    logs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return x, h, np.exp(logs), logs


def _policy_backward(p, x, h, g_logits, prefix=''):
    g_h = (g_logits@p[prefix+'pi'].T)*(1 - h*h)
    return {prefix+'pi': h.T@g_logits, prefix+'pb': g_logits.sum(axis=0),
            prefix+'w': x.T@g_h, prefix+'b': g_h.sum(axis=0)}


def _q_init(rng, obs_size, hidden, actions, prefix):
    """Same initialisation as the Double DQN network in agents.py."""
    return {prefix+'w1': rng.normal(0, np.sqrt(2/obs_size), (obs_size, hidden)),
            prefix+'b1': np.zeros(hidden),
            prefix+'w2': rng.normal(0, np.sqrt(2/hidden), (hidden, hidden)),
            prefix+'b2': np.zeros(hidden),
            prefix+'wq': rng.normal(0, .01/np.sqrt(hidden), (hidden, actions)),
            prefix+'bq': np.zeros(actions)}


def _q_forward(p, x, prefix):
    h1 = np.maximum(x@p[prefix+'w1'] + p[prefix+'b1'], 0.)
    h2 = np.maximum(h1@p[prefix+'w2'] + p[prefix+'b2'], 0.)
    return h1, h2, h2@p[prefix+'wq'] + p[prefix+'bq']


def _q_backward(p, x, h1, h2, gq, prefix):
    gh2 = (gq@p[prefix+'wq'].T)*(h2 > 0)
    gh1 = (gh2@p[prefix+'w2'].T)*(h1 > 0)
    return {prefix+'wq': h2.T@gq, prefix+'bq': gq.sum(axis=0),
            prefix+'w2': h1.T@gh2, prefix+'b2': gh2.sum(axis=0),
            prefix+'w1': x.T@gh1, prefix+'b1': gh1.sum(axis=0)}


def _choose(rng, probs, deterministic):
    """Argmax, or one categorical draw per row (the sampler PPO uses)."""
    if deterministic:
        return probs.argmax(axis=1)
    draws = rng.random(len(probs))[:, None]
    return (draws > probs.cumsum(axis=1)).sum(axis=1).clip(max=probs.shape[1]-1)


class _NumpyAgent:
    """Adam and checkpointing shared by the agents in this module."""
    state = {'p': 'params', 'm': 'm', 's': 's'}   # checkpoint prefix -> parameter dict attribute

    def _init_optimizer(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.s = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.updates = 0

    def _adam(self, grads, lr, max_norm=None):
        """Adam step as in agents.py on the parameters in `grads`; increment self.updates first."""
        norm = float(np.sqrt(sum(np.square(g).sum() for g in grads.values())))
        scale = min(1., max_norm/(norm + 1e-8)) if max_norm else 1.
        for key, grad in grads.items():
            grad = grad*scale
            self.m[key] = .9*self.m[key] + .1*grad
            self.s[key] = .999*self.s[key] + .001*grad*grad
            mh = self.m[key]/(1 - .9**self.updates)
            sh = self.s[key]/(1 - .999**self.updates)
            self.params[key] -= lr*mh/(np.sqrt(sh) + 1e-8)
        return norm

    def _check_finite(self):
        if not all(np.isfinite(v).all() for v in self.params.values()):
            raise FloatingPointError(f'{type(self).__name__} parameters became non-finite')

    def _extra_state(self):
        return {}

    def _load_extra(self, data):
        pass

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(config=np.array(json.dumps(self.config)),
                       metadata=np.array(json.dumps(metadata or {})),
                       rng=np.array(json.dumps(self.rng.bit_generator.state)),
                       updates=np.array(self.updates))
        for prefix, name in self.state.items():
            payload.update({f'{prefix}_{k}': v for k, v in getattr(self, name).items()})
        payload.update(self._extra_state())
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            agent = cls(**json.loads(str(data['config'])))
            for prefix, name in cls.state.items():
                values = getattr(agent, name)
                for key in values:
                    values[key] = data[f'{prefix}_{key}'].copy()
            agent.updates = int(data['updates'])
            agent.rng.bit_generator.state = json.loads(str(data['rng']))
            agent.metadata = json.loads(str(data['metadata']))
            agent._load_extra(data)
        return agent


class DiscreteSAC(_NumpyAgent):
    """Discrete-action SAC: categorical actor, twin Q critics with Polyak-averaged
    targets, exact expectations over actions, learned entropy temperature and
    uniform replay. Only true terminations stop bootstrapping."""
    state = {'p': 'params', 't': 'target', 'm': 'm', 's': 's'}
    observe = DoubleDQN.observe   # the same uniform replay ring buffer

    def __init__(self, obs_size=8, actions=4, hidden=64, seed=0, lr=3e-4, gamma=.99, tau=.005,
                 capacity=50000, target_entropy_fraction=.5, initial_alpha=1.):
        self.config = dict(obs_size=obs_size, actions=actions, hidden=hidden, seed=seed, lr=lr,
                           gamma=gamma, tau=tau, capacity=capacity,
                           target_entropy_fraction=target_entropy_fraction, initial_alpha=initial_alpha)
        self.rng = np.random.default_rng(seed)
        self.lr, self.gamma, self.tau, self.capacity = lr, gamma, tau, capacity
        # Constant target entropy: every state has the same four valid actions.
        self.target_entropy = float(target_entropy_fraction*np.log(actions))
        self.params = _policy_init(self.rng, obs_size, hidden, actions, 'a_')
        self.params.update(_q_init(self.rng, obs_size, hidden, actions, 'q1_'))
        self.params.update(_q_init(self.rng, obs_size, hidden, actions, 'q2_'))
        self.params['log_alpha'] = np.array([np.log(initial_alpha)])
        self.critic_keys = [k for k in self.params if k.startswith('q')]
        self.target = {k: self.params[k].copy() for k in self.critic_keys}
        self._init_optimizer()
        self.size = self.position = 0
        self.replay_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.replay_next = np.zeros_like(self.replay_obs)
        self.replay_actions = np.zeros(capacity, dtype=np.int64)
        self.replay_rewards = np.zeros(capacity, dtype=np.float32)
        self.replay_terminated = np.zeros(capacity, dtype=bool)

    @property
    def alpha(self):
        return float(np.exp(self.params['log_alpha'][0]))

    def policy(self, obs):
        return _policy_forward(self.params, obs, 'a_')

    def q_values(self, obs, target=False):
        x = np.atleast_2d(obs).astype(np.float64)
        p = self.target if target else self.params
        return _q_forward(p, x, 'q1_')[2], _q_forward(p, x, 'q2_')[2]

    def act(self, obs, deterministic=False):
        _, _, probs, logs = self.policy(obs)
        actions = _choose(self.rng, probs, deterministic)
        return actions, logs[np.arange(len(actions)), actions]

    def critic_targets(self, rewards, next_obs, terminated):
        """r + gamma * sum_a' pi(a'|s') [min_i Q_i^target(s',a') - alpha log pi(a'|s')], exactly."""
        _, _, probs, logs = self.policy(next_obs)
        q = np.minimum(*self.q_values(next_obs, target=True))
        soft_value = (probs*(q - self.alpha*logs)).sum(axis=1)
        return (np.asarray(rewards, dtype=np.float64)
                + self.gamma*(~np.asarray(terminated, dtype=bool))*soft_value)

    def critic_loss_grad(self, obs, actions, targets):
        x = np.atleast_2d(obs).astype(np.float64)
        rows, actions = np.arange(len(x)), np.asarray(actions)
        grads, loss, chosen = {}, 0., []
        for prefix in ('q1_', 'q2_'):
            h1, h2, q = _q_forward(self.params, x, prefix)
            error = q[rows, actions] - targets
            loss += .5*np.mean(error*error)
            gq = np.zeros_like(q)
            gq[rows, actions] = error/len(x)
            grads.update(_q_backward(self.params, x, h1, h2, gq, prefix))
            chosen.append(q[rows, actions])
        return dict(critic_loss=float(loss), mean_q=float(np.mean(chosen))), grads

    def actor_loss_grad(self, obs):
        """Loss E_s sum_a pi(a|s) [alpha log pi(a|s) - min_i Q_i(s,a)]; critics are held fixed."""
        x, h, probs, logs = self.policy(obs)
        q = np.minimum(*self.q_values(x))
        inner = self.alpha*logs - q
        expected = (probs*inner).sum(axis=1)
        g_logits = probs*(inner - expected[:, None])/len(x)
        entropy = -(probs*logs).sum(axis=1)
        grads = _policy_backward(self.params, x, h, g_logits, 'a_')
        return dict(actor_loss=float(expected.mean()), entropy=float(entropy.mean())), grads, entropy

    def train_batch(self, obs, actions, rewards, next_obs, terminated):
        """One critic, actor and temperature step on a batch, then a soft target update."""
        alpha = self.alpha
        targets = self.critic_targets(rewards, next_obs, terminated)
        critic_metrics, critic_grads = self.critic_loss_grad(obs, actions, targets)
        self.updates += 1
        self._adam(critic_grads, self.lr, 10.)
        actor_metrics, actor_grads, entropy = self.actor_loss_grad(obs)
        self._adam(actor_grads, self.lr, 10.)
        # Temperature loss alpha * (H(pi) - target): alpha falls while entropy is above target.
        self._adam({'log_alpha': np.array([alpha*np.mean(entropy - self.target_entropy)])}, self.lr)
        for k in self.critic_keys:
            self.target[k] += self.tau*(self.params[k] - self.target[k])
        self._check_finite()
        return dict(**critic_metrics, **actor_metrics, alpha=alpha)

    def update(self, batch_size=64):
        if self.size < batch_size:
            return dict(critic_loss=0., mean_q=0., actor_loss=0., entropy=0., alpha=self.alpha, skipped=1)
        idx = self.rng.integers(self.size, size=batch_size)
        return self.train_batch(self.replay_obs[idx], self.replay_actions[idx], self.replay_rewards[idx],
                                self.replay_next[idx], self.replay_terminated[idx])

    def _extra_state(self):
        # Replay is saved, as for Double DQN, so a reloaded agent can continue exactly.
        state = dict(size=np.array(self.size), position=np.array(self.position))
        for key in ('obs', 'next', 'actions', 'rewards', 'terminated'):
            state['replay_'+key] = getattr(self, 'replay_'+key)[:self.size]
        return state

    def _load_extra(self, data):
        self.size, self.position = int(data['size']), int(data['position'])
        for key in ('obs', 'next', 'actions', 'rewards', 'terminated'):
            getattr(self, 'replay_'+key)[:self.size] = data['replay_'+key]


class GRPO(_NumpyAgent):
    """Group-relative policy optimisation without a critic (see the module docstring)."""
    state = {'p': 'params', 'r': 'reference', 'm': 'm', 's': 's'}

    def __init__(self, obs_size=8, actions=4, hidden=64, seed=0, lr=3e-4, clip=.2, kl_coef=.04,
                 group_size=8, epochs=4, batch_size=128, target_kl=.03):
        self.config = dict(obs_size=obs_size, actions=actions, hidden=hidden, seed=seed, lr=lr,
                           clip=clip, kl_coef=kl_coef, group_size=group_size, epochs=epochs,
                           batch_size=batch_size, target_kl=target_kl)
        self.rng = np.random.default_rng(seed)
        self.lr, self.clip, self.kl_coef = lr, clip, kl_coef
        self.epochs, self.batch_size, self.target_kl = epochs, batch_size, target_kl
        self.params = _policy_init(self.rng, obs_size, hidden, actions)
        self.reference = {k: v.copy() for k, v in self.params.items()}   # fixed initial policy
        self._init_optimizer()

    def policy(self, obs, reference=False):
        return _policy_forward(self.reference if reference else self.params, obs)

    def act(self, obs, deterministic=False):
        _, _, probs, logs = self.policy(obs)
        actions = _choose(self.rng, probs, deterministic)
        return actions, logs[np.arange(len(actions)), actions]

    @staticmethod
    def group_advantages(returns, complete=None, min_std=1e-8):
        """(R_i - mean) / std over one group, or (None, reason) when the group must be skipped."""
        returns = np.asarray(returns, dtype=np.float64)
        if complete is not None and not np.all(complete):
            return None, 'incomplete'
        std = returns.std()
        if len(returns) < 2 or not std > min_std:
            return None, 'zero_variance'
        return (returns - returns.mean())/std, None

    def loss_grad(self, obs, actions, old_logs, advantages, reference_logs):
        x, h, probs, logs = self.policy(obs)
        n, actions = len(x), np.asarray(actions)
        rows = np.arange(n)
        ratio = np.exp(np.clip(logs[rows, actions] - old_logs, -60, 60))
        clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip)
        surrogate = np.minimum(ratio*advantages, clipped*advantages)
        active = ((advantages >= 0) & (ratio <= 1 + self.clip)) | ((advantages < 0) & (ratio >= 1 - self.clip))
        g_logits = (-advantages*ratio*active/n)[:, None]*(np.eye(probs.shape[1])[actions] - probs)
        # Exact KL(pi || pi_ref) over the four actions and its gradient.
        difference = logs - reference_logs
        kl = (probs*difference).sum(axis=1)
        g_logits += self.kl_coef/n*probs*(difference - kl[:, None])
        grads = _policy_backward(self.params, x, h, g_logits)
        metrics = dict(loss=float(-surrogate.mean() + self.kl_coef*kl.mean()),
                       kl_to_reference=float(kl.mean()),
                       entropy=float(-(probs*logs).sum(axis=1).mean()),
                       approx_kl=float(((ratio - 1) - (logs[rows, actions] - old_logs)).mean()),
                       clip_fraction=float((np.abs(ratio - 1) > self.clip).mean()))
        return metrics, grads

    def update_group(self, episodes):
        """episodes: dicts with obs, actions, logs (sampling log-probs), return_ and complete."""
        returns = np.array([e['return_'] for e in episodes], dtype=np.float64)
        advantages, reason = self.group_advantages(returns, [e['complete'] for e in episodes])
        info = dict(group_episodes=len(episodes), group_return_mean=float(returns.mean()),
                    group_return_std=float(returns.std()),
                    group_transitions=int(sum(len(e['actions']) for e in episodes)))
        if advantages is None:
            return dict(info, skipped=1, skip_reason=reason)
        obs = np.concatenate([np.asarray(e['obs'], dtype=np.float64).reshape(-1, self.config['obs_size'])
                              for e in episodes])
        actions = np.concatenate([np.asarray(e['actions'], dtype=np.int64) for e in episodes])
        old_logs = np.concatenate([np.asarray(e['logs'], dtype=np.float64) for e in episodes])
        per_action = np.repeat(advantages, [len(e['actions']) for e in episodes])
        reference_logs = self.policy(obs, reference=True)[3]
        history, stopped = [], False
        for _ in range(self.epochs):
            order = self.rng.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                idx = order[start:start + self.batch_size]
                metrics, grads = self.loss_grad(obs[idx], actions[idx], old_logs[idx],
                                                per_action[idx], reference_logs[idx])
                history.append(metrics)
                if metrics['approx_kl'] > self.target_kl:
                    stopped = True
                    break
                self.updates += 1
                self._adam(grads, self.lr, .5)
                self._check_finite()
            if stopped:
                break
        means = {k: float(np.mean([m[k] for m in history])) for k in history[0]}
        return dict(info, skipped=0, early_stop=int(stopped), **means)


class GFlowNetTB(_NumpyAgent):
    """GFlowNet forward policy trained with trajectory balance (see the module docstring)."""

    def __init__(self, obs_size=8, actions=4, hidden=64, seed=0, lr=3e-4, logz_lr=1.,
                 beta=float(1000*np.log(4)), return_min=-400., return_max=300., best_size=32,
                 latest_size=32, batch_trajectories=16, updates_per_rollout=32):
        self.config = dict(obs_size=obs_size, actions=actions, hidden=hidden, seed=seed, lr=lr,
                           logz_lr=logz_lr, beta=beta, return_min=return_min, return_max=return_max,
                           best_size=best_size, latest_size=latest_size,
                           batch_trajectories=batch_trajectories, updates_per_rollout=updates_per_rollout)
        self.rng = np.random.default_rng(seed)
        self.lr, self.logz_lr = lr, logz_lr
        self.params = _policy_init(self.rng, obs_size, hidden, actions)
        self.params['log_z'] = np.zeros(1)
        self._init_optimizer()
        self.logz_initialised = False
        self.best, self.latest = [], []

    def policy(self, obs):
        return _policy_forward(self.params, obs)

    def act(self, obs, deterministic=False):
        _, _, probs, logs = self.policy(obs)
        actions = _choose(self.rng, probs, deterministic)
        return actions, logs[np.arange(len(actions)), actions]

    def log_reward(self, returns):
        c = self.config
        scaled = (np.asarray(returns, dtype=np.float64) - c['return_min'])/(c['return_max'] - c['return_min'])
        return c['beta']*np.clip(scaled, 0., 1.)

    def trajectory(self, obs, actions, return_):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, self.config['obs_size'])
        actions = np.asarray(actions, dtype=np.int64)
        if len(obs) != len(actions) or not len(obs):
            raise ValueError('A trajectory needs one action per visited state')
        return dict(obs=obs, actions=actions, return_=float(return_),
                    log_reward=float(self.log_reward(return_)))

    def add_trajectory(self, obs, actions, return_):
        """Store one complete episode in the latest-K and best-K (by return) replay pools."""
        trajectory = self.trajectory(obs, actions, return_)
        self.latest = (self.latest + [trajectory])[-self.config['latest_size']:]
        # Stable sort: among equal returns the earlier trajectory is kept.
        self.best = sorted(self.best + [trajectory], key=lambda t: -t['return_'])[:self.config['best_size']]

    def tb_loss_grad(self, trajectories):
        lengths = np.array([len(t['actions']) for t in trajectories])
        obs = np.concatenate([t['obs'] for t in trajectories])
        actions = np.concatenate([t['actions'] for t in trajectories])
        log_r = np.array([t['log_reward'] for t in trajectories], dtype=np.float64)
        segment = np.repeat(np.arange(len(trajectories)), lengths)
        x, h, probs, logs = self.policy(obs)
        sum_log_pf = np.bincount(segment, weights=logs[np.arange(len(actions)), actions],
                                 minlength=len(trajectories))
        # Tree-shaped trajectory space: every state has one parent, so log P_B = 0.
        delta = self.params['log_z'][0] + sum_log_pf - log_r
        weight = (2*delta/len(trajectories))[segment]
        g_logits = weight[:, None]*(np.eye(probs.shape[1])[actions] - probs)
        grads = _policy_backward(self.params, x, h, g_logits)
        grads['log_z'] = np.array([2*delta.mean()])
        metrics = dict(tb_loss=float(np.mean(delta*delta)), log_z=float(self.params['log_z'][0]),
                       mean_abs_residual=float(np.abs(delta).mean()), mean_log_reward=float(log_r.mean()),
                       mean_sum_log_pf=float(sum_log_pf.mean()), mean_length=float(lengths.mean()),
                       entropy=float(-(probs*logs).sum(axis=1).mean()))
        return metrics, grads, delta

    def sample_batch(self):
        half, batch = self.config['batch_trajectories']//2, []
        for pool in (self.latest, self.best):
            k = min(half, len(pool))
            batch += [pool[i] for i in self.rng.choice(len(pool), size=k, replace=False)]
        return batch

    def update(self):
        if not self.latest:
            return dict(tb_loss=0., skipped=1)
        history = []
        for _ in range(self.config['updates_per_rollout']):
            batch = self.sample_batch()
            if not self.logz_initialised:
                # Start log Z at mean(log R - sum log P_F), its least-squares optimum for this batch.
                self.params['log_z'] -= self.tb_loss_grad(batch)[2].mean()
                self.logz_initialised = True
            metrics, grads, _ = self.tb_loss_grad(batch)
            logz_grad = {'log_z': grads.pop('log_z')}
            self.updates += 1
            self._adam(grads, self.lr, .5)
            self._adam(logz_grad, self.logz_lr)
            self._check_finite()
            history.append(metrics)
        means = {k: float(np.mean([m[k] for m in history])) for k in history[0]}
        return dict(means, skipped=0, best_replay_return=self.best[0]['return_'],
                    replay_best=len(self.best), replay_latest=len(self.latest))

    def _extra_state(self):
        state = {'logz_initialised': np.array(self.logz_initialised)}
        for name in ('best', 'latest'):
            pool = getattr(self, name)
            state[f'replay_{name}_lengths'] = np.array([len(t['actions']) for t in pool], dtype=np.int64)
            state[f'replay_{name}_obs'] = (np.concatenate([t['obs'] for t in pool]) if pool
                                           else np.zeros((0, self.config['obs_size']), dtype=np.float32))
            state[f'replay_{name}_actions'] = (np.concatenate([t['actions'] for t in pool]) if pool
                                               else np.zeros(0, dtype=np.int64))
            state[f'replay_{name}_returns'] = np.array([t['return_'] for t in pool], dtype=np.float64)
        return state

    def _load_extra(self, data):
        self.logz_initialised = bool(data['logz_initialised'])
        for name in ('best', 'latest'):
            splits = np.cumsum(data[f'replay_{name}_lengths'])[:-1]
            parts = zip(np.split(data[f'replay_{name}_obs'], splits),
                        np.split(data[f'replay_{name}_actions'], splits), data[f'replay_{name}_returns'])
            setattr(self, name, [self.trajectory(o, a, r) for o, a, r in parts])
