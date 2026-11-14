import tempfile
import unittest
from pathlib import Path
import numpy as np
from lunar_bench.agents import PPO,A2C,DoubleDQN,gae
from lunar_bench.agents_extra import DiscreteSAC,GRPO,GFlowNetTB


class AgentTests(unittest.TestCase):
    def finite_differences(self,agent,loss,grads,keys=None):
        for key in (agent.params if keys is None else keys):
            p=agent.params[key]
            for idx in list(np.ndindex(p.shape))[:5]:
                old=p[idx];eps=1e-6
                p[idx]=old+eps;plus=loss()
                p[idx]=old-eps;minus=loss()
                p[idx]=old
                self.assertAlmostEqual(float(grads[key][idx]),(plus-minus)/(2*eps),places=5,msg=f'{key}{idx}')

    def test_a2c_gradients_and_single_update(self):
        agent=A2C(obs_size=3,actions=2,hidden=4,seed=7)
        rng=np.random.default_rng(4)
        obs=rng.normal(size=(7,3));actions=rng.integers(2,size=7)
        args=(obs,actions,np.zeros(7),rng.normal(size=7),rng.normal(size=7))
        _,grads=agent.loss_grad(*args)
        self.finite_differences(agent,lambda:agent.loss_grad(*args)[0]['loss'],grads)
        agent.update(*args,epochs=10,batch_size=1)
        self.assertEqual(agent.updates,1)

    def test_dqn_huber_gradients(self):
        agent=DoubleDQN(obs_size=3,actions=2,hidden=4,seed=7)
        rng=np.random.default_rng(1)
        obs=rng.normal(size=(7,3));actions=rng.integers(2,size=7)
        targets=np.array([.1,-.3,2.,-2.,.6,.2,3.])
        _,grads=agent.loss_grad(obs,actions,targets)
        self.finite_differences(agent,lambda:agent.loss_grad(obs,actions,targets)[0]['loss'],grads)

    def test_double_target_and_terminal_bootstrap(self):
        agent=DoubleDQN(obs_size=2,actions=2,hidden=4)
        for params in (agent.params,agent.target):
            for p in params.values():p.fill(0)
        agent.params['bq'][:]=[5.,0.]
        agent.target['bq'][:]=[1.,3.]
        target=agent.targets([2.,2.],np.zeros((2,2)),[True,False])
        np.testing.assert_allclose(target,[2.,2.+.99])

    def test_dqn_replay_wrap_and_checkpoint(self):
        agent=DoubleDQN(obs_size=2,actions=2,hidden=4,capacity=5,target_interval=2)
        obs=np.arange(16,dtype=float).reshape(8,2)/16
        agent.observe(obs[:4],[0]*4,[1]*4,obs[1:5],[False]*4)
        agent.observe(obs[4:],[1]*4,[2]*4,obs[4:],[True]*4)
        self.assertEqual((agent.size,agent.position),(5,3))
        agent.update(3);agent.update(3)
        for k in agent.params:np.testing.assert_array_equal(agent.params[k],agent.target[k])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'dqn.npz';agent.save(path,{'test':True})
            loaded=DoubleDQN.load(path)
            self.assertEqual(loaded.metadata,{'test':True})
            self.assertEqual(agent.update(3),loaded.update(3))
            for k in agent.params:np.testing.assert_array_equal(agent.params[k],loaded.params[k])

    def test_policy_checkpoint_and_deterministic_rng(self):
        for cls in (PPO,A2C):
            agent=cls(seed=3)
            state=agent.rng.bit_generator.state
            agent.act(np.zeros((5,8)),deterministic=True)
            self.assertEqual(agent.rng.bit_generator.state,state)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.npz';agent.save(path)
                loaded=cls.load(path)
                np.testing.assert_array_equal(agent.act(np.zeros((5,8)))[0],loaded.act(np.zeros((5,8)))[0])

    def test_gae_time_limit_and_reset_boundary(self):
        rewards=np.array([[1.],[100.]])
        values=np.zeros_like(rewards)
        next_values=np.array([[10.],[0.]])
        terminated=np.array([[False],[True]])
        ended=np.ones_like(terminated)
        adv,_=gae(rewards,values,next_values,terminated,ended)
        np.testing.assert_allclose(adv[:,0],[10.9,100.])

    def test_dqn_contextual_bandit(self):
        agent=DoubleDQN(obs_size=2,actions=2,hidden=16,seed=2,lr=.003)
        rng=np.random.default_rng(7)
        states=rng.normal(size=(500,2))
        actions=rng.integers(2,size=500)
        reward=(actions==(states[:,0]>0)).astype(float)
        agent.observe(states,actions,reward,states,np.ones(500,dtype=bool))
        for _ in range(400):agent.update(64)
        test=rng.normal(size=(500,2))
        self.assertGreater(np.mean(agent.act(test)==(test[:,0]>0)),.9)

    def test_new_agents_shapes_probabilities_and_greedy_rng(self):
        obs=np.random.default_rng(0).normal(size=(5,8))
        for cls in (DiscreteSAC,GRPO,GFlowNetTB):
            agent=cls(seed=3)
            probs=agent.policy(obs)[2]
            self.assertEqual(probs.shape,(5,4))
            self.assertTrue(np.isfinite(probs).all())
            np.testing.assert_allclose(probs.sum(axis=1),1.)
            state=agent.rng.bit_generator.state
            greedy,logs=agent.act(obs,deterministic=True)
            self.assertEqual(agent.rng.bit_generator.state,state)
            np.testing.assert_array_equal(greedy,probs.argmax(axis=1))
            actions,logs=agent.act(obs)
            self.assertEqual(actions.shape,(5,))
            self.assertTrue(((actions>=0)&(actions<4)).all() and np.isfinite(logs).all())
            np.testing.assert_allclose(logs,np.log(probs[np.arange(5),actions]))
        for q in DiscreteSAC(seed=1).q_values(obs):
            self.assertEqual(q.shape,(5,4))
            self.assertTrue(np.isfinite(q).all())

    def test_sac_gradients_and_exact_soft_target(self):
        agent=DiscreteSAC(obs_size=3,actions=2,hidden=4,seed=7,initial_alpha=.5)
        rng=np.random.default_rng(2)
        obs=rng.normal(size=(7,3));actions=rng.integers(2,size=7);targets=rng.normal(size=7)
        _,grads=agent.critic_loss_grad(obs,actions,targets)
        self.finite_differences(agent,lambda:agent.critic_loss_grad(obs,actions,targets)[0]['critic_loss'],grads,keys=grads)
        _,grads,_=agent.actor_loss_grad(obs)
        self.finite_differences(agent,lambda:agent.actor_loss_grad(obs)[0]['actor_loss'],grads,keys=grads)
        for params in (agent.params,agent.target):
            for p in params.values():p.fill(0)
        agent.target['q1_bq'][:]=[1.,3.];agent.target['q2_bq'][:]=[2.,0.]
        target=agent.critic_targets([1.,1.],np.zeros((2,3)),[True,False])
        # Uniform policy, alpha=1: E[min(Q1,Q2)] = (1+0)/2 plus entropy log 2, with no sampling.
        np.testing.assert_allclose(target,[1.,1.+.99*(.5+np.log(2))])

    def test_sac_update_lowers_critic_loss_on_fixed_batch(self):
        agent=DiscreteSAC(obs_size=3,actions=2,hidden=8,seed=5,lr=1e-3)
        rng=np.random.default_rng(3)
        obs,next_obs=rng.normal(size=(32,3)),rng.normal(size=(32,3))
        actions,rewards,terminated=rng.integers(2,size=32),rng.normal(size=32),rng.random(32)<.3
        targets=agent.critic_targets(rewards,next_obs,terminated)
        before=agent.critic_loss_grad(obs,actions,targets)[0]['critic_loss']
        old_target={k:v.copy() for k,v in agent.target.items()};old_alpha=agent.alpha
        agent.train_batch(obs,actions,rewards,next_obs,terminated)
        self.assertLess(agent.critic_loss_grad(obs,actions,targets)[0]['critic_loss'],before)
        for k in agent.target:
            np.testing.assert_allclose(agent.target[k],old_target[k]+agent.tau*(agent.params[k]-old_target[k]))
        self.assertLess(agent.alpha,old_alpha)   # near-uniform policy is above the target entropy

    def test_grpo_group_advantages_and_skipped_groups(self):
        advantages,reason=GRPO.group_advantages([1.,2.,3.,6.])
        self.assertIsNone(reason)
        self.assertAlmostEqual(advantages.mean(),0.,places=12)
        self.assertAlmostEqual(advantages.std(),1.,places=12)
        self.assertEqual(int(advantages.argmax()),3)
        self.assertEqual(GRPO.group_advantages([5.,5.,5.]),(None,'zero_variance'))
        self.assertEqual(GRPO.group_advantages([1.,2.],complete=[True,False]),(None,'incomplete'))
        agent=GRPO(obs_size=3,actions=2,hidden=4,seed=1,group_size=3)
        state=np.ones((4,3))
        def episode(action,score,complete=True):
            logs=agent.policy(state)[3][:,action]
            return dict(obs=state,actions=[action]*4,logs=logs,return_=score,complete=complete)
        before={k:v.copy() for k,v in agent.params.items()}
        for group in ([episode(0,5.),episode(1,5.),episode(0,5.)],[episode(0,9.),episode(1,1.),episode(1,2.,False)]):
            self.assertEqual(agent.update_group(group)['skipped'],1)
        self.assertEqual(agent.updates,0)
        for k in before:np.testing.assert_array_equal(agent.params[k],before[k])
        old=agent.policy(state[:1])[2][0,0]
        metrics=agent.update_group([episode(0,10.),episode(1,0.),episode(1,1.)])
        self.assertEqual(metrics['skipped'],0)
        self.assertGreater(agent.policy(state[:1])[2][0,0],old)   # best episode's action gains probability

    def test_grpo_gradients(self):
        agent=GRPO(obs_size=3,actions=2,hidden=4,seed=7,kl_coef=.3)
        rng=np.random.default_rng(5)
        for v in agent.reference.values():v+=rng.normal(scale=.3,size=v.shape)   # nonzero KL term
        obs=rng.normal(size=(7,3));actions=rng.integers(2,size=7)
        old=agent.policy(obs)[3][np.arange(7),actions]+rng.uniform(-.1,.1,size=7)   # ratios inside the clip range
        args=(obs,actions,old,rng.normal(size=7),agent.policy(obs,reference=True)[3])
        _,grads=agent.loss_grad(*args)
        self.finite_differences(agent,lambda:agent.loss_grad(*args)[0]['loss'],grads)

    def test_gflownet_tb_loss_zero_for_consistent_case(self):
        agent=GFlowNetTB(obs_size=3,actions=4,hidden=4,beta=10.,return_min=0.,return_max=10.)
        for p in agent.params.values():p.fill(0)   # uniform forward policy: log P_F = -log 4 per step
        agent.params['log_z'][:]=5.
        rng=np.random.default_rng(0)
        # With beta=10 on [0,10], log R equals the return; choose log R = log Z + sum log P_F.
        trajectories=[agent.trajectory(rng.normal(size=(n,3)),rng.integers(4,size=n),5.-n*np.log(4)) for n in (1,2,3)]
        metrics,grads,residual=agent.tb_loss_grad(trajectories)
        self.assertAlmostEqual(metrics['tb_loss'],0.,places=20)
        np.testing.assert_allclose(residual,0.,atol=1e-12)
        for g in grads.values():np.testing.assert_allclose(g,0.,atol=1e-12)
        metrics,grads,_=agent.tb_loss_grad([agent.trajectory(np.zeros((2,3)),[0,1],1.)])
        expected=5.-2*np.log(4)-1.
        self.assertAlmostEqual(metrics['tb_loss'],expected**2)
        self.assertAlmostEqual(float(grads['log_z'][0]),2*expected)

    def test_gflownet_gradients_reward_map_and_replay(self):
        agent=GFlowNetTB(obs_size=3,actions=2,hidden=4,seed=7,beta=5.,best_size=3,latest_size=2)
        agent.params['log_z'][:]=.7
        rng=np.random.default_rng(6)
        trajectories=[agent.trajectory(rng.normal(size=(n,3)),rng.integers(2,size=n),r) for n,r in ((2,-100.),(4,50.),(3,280.))]
        _,grads,_=agent.tb_loss_grad(trajectories)
        self.finite_differences(agent,lambda:agent.tb_loss_grad(trajectories)[0]['tb_loss'],grads)
        np.testing.assert_allclose(agent.log_reward([-1000.,-400.,-50.,300.,900.]),[0.,0.,2.5,5.,5.])
        for r in (5.,1.,9.,3.,7.):agent.add_trajectory(np.zeros((2,3)),[0,1],r)
        self.assertEqual([t['return_'] for t in agent.best],[9.,7.,5.])
        self.assertEqual([t['return_'] for t in agent.latest],[3.,7.])

    def test_new_agents_checkpoint_round_trip(self):
        rng=np.random.default_rng(9)
        obs=rng.normal(size=(40,8))
        sac=DiscreteSAC(seed=1,capacity=30)
        sac.observe(obs,rng.integers(4,size=40),rng.normal(size=40),rng.normal(size=(40,8)),rng.random(40)<.2)
        sac.update(16)
        grpo=GRPO(seed=2,group_size=4)
        episodes=[]
        for i in range(4):
            actions,logs=grpo.act(obs[5*i:5*i+5])
            episodes.append(dict(obs=obs[5*i:5*i+5],actions=actions,logs=logs,return_=float(i),complete=True))
        grpo.update_group(episodes)
        gfn=GFlowNetTB(seed=3)
        for i in range(4):gfn.add_trajectory(obs[5*i:5*i+5],rng.integers(4,size=5),float(rng.normal(0,100)))
        gfn.update()
        continue_training={DiscreteSAC:lambda a:a.update(16),GRPO:lambda a:a.update_group(episodes),
                           GFlowNetTB:lambda a:a.update()}
        for agent in (sac,grpo,gfn):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'agent.npz';agent.save(path,{'test':True})
                loaded=type(agent).load(path)
            self.assertEqual(loaded.metadata,{'test':True})
            self.assertEqual(loaded.updates,agent.updates)
            for name in agent.state.values():
                for k,v in getattr(agent,name).items():np.testing.assert_array_equal(v,getattr(loaded,name)[k])
            np.testing.assert_array_equal(agent.act(obs)[0],loaded.act(obs)[0])
            step=continue_training[type(agent)]
            self.assertEqual(step(agent),step(loaded))
            for k in agent.params:np.testing.assert_array_equal(agent.params[k],loaded.params[k])

if __name__=='__main__':unittest.main()
