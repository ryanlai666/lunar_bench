import tempfile
import unittest
from pathlib import Path
import numpy as np
from lunar_bench.agents import PPO,A2C,DoubleDQN,gae


class AgentTests(unittest.TestCase):
    def finite_differences(self,agent,loss,grads):
        for key in agent.params:
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

if __name__=='__main__':unittest.main()
