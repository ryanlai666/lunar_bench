import ctypes as ct
import unittest
import numpy as np
from lunar_bench.rust_env import RustEnv,RustBatchEnv,heuristic

class RustEnvTests(unittest.TestCase):
    def test_scalar_batch_match(self):
        with RustEnv(horizon=50) as scalar,RustBatchEnv(2,horizon=50) as batch:
            s,_=scalar.reset(seed=123);b=batch.reset_at(0,123);batch.reset_at(1,456)
            np.testing.assert_array_equal(s,b)
            for i in range(50):
                s,r,t,u,_=scalar.step(i%4);b,br,bt,bu=batch.step([i%4,(i+1)%4])
                np.testing.assert_array_equal(s,b[0]);self.assertEqual(r,float(br[0]));self.assertEqual(t,bool(bt[0]));self.assertEqual(u,bool(bu[0]))
            self.assertTrue(u);self.assertFalse(t)
            with self.assertRaises(RuntimeError):scalar.step(0)
    def test_lifecycle_and_validation(self):
        e=RustEnv()
        with self.assertRaises(RuntimeError):e.step(0)
        with self.assertRaises(ValueError):e.reset(seed=-1)
        e.reset(seed=0)
        for a in [-1,4,0.5,float('nan'),True]:
            with self.assertRaises(ValueError):e.step(a)
        e.close();e.close()
        with self.assertRaises(RuntimeError):e.reset(seed=0)
    def test_native_invalid_batch_is_atomic(self):
        with RustBatchEnv(2) as a,RustBatchEnv(2) as b:
            for e in (a,b):
                for i in range(2):e.reset_at(i,i)
            bad=np.array([2,4],dtype=np.uint32)
            code=a.lib.rl_step(a.handle,bad.ctypes.data_as(ct.POINTER(ct.c_uint32)),a.output.ctypes.data_as(ct.POINTER(ct.c_float)))
            self.assertEqual(code,-2)
            a.step([1,2]);b.step([1,2]);np.testing.assert_array_equal(a.output,b.output)
    def test_heuristic_landings(self):
        # This checks that the implemented dynamics admit controlled safe touchdowns.
        success=0
        with RustEnv() as e:
            for seed in range(20):
                obs,_=e.reset(seed=seed)
                while True:
                    obs,r,t,u,info=e.step(heuristic(obs))
                    self.assertTrue(np.isfinite(obs).all())
                    if t or u:break
                success+=int(info['success'])
        self.assertGreaterEqual(success,15)

if __name__=='__main__':unittest.main()
