"""Minimal invariant tests for the independent QGPF implementation."""
import unittest
import numpy as np
from qgpf import FilterConfig, ModelConfig, decode_state, encode_state, simulate_truth, run_filter

class QGPFTests(unittest.TestCase):
    def test_binary_roundtrip_is_bounded(self):
        cfg=FilterConfig(); x=np.array([0.1,-2.0,30.0,5.0]); bits=encode_state(x,cfg); y=decode_state(bits,cfg)
        self.assertTrue(np.all(y>=np.asarray(cfg.state_min))); self.assertTrue(np.all(y<=np.asarray(cfg.state_max)))
    def test_simulation_shapes(self):
        t,z=simulate_truth(np.random.default_rng(4),ModelConfig(horizon=7)); self.assertEqual(t.shape,(8,4)); self.assertEqual(z.shape,(8,))
    def test_filter_finite(self):
        m=ModelConfig(horizon=8); c=FilterConfig(particle_count=30,generation_count=2); t,z=simulate_truth(np.random.default_rng(2),m); r=run_filter(z,t,np.random.default_rng(3),"qgpf",m,c)
        for key in ("rmse_position","rmse_velocity","rmse_state","mean_ess","mean_distinct","runtime_s"): self.assertTrue(np.isfinite(r[key]),key)

if __name__=="__main__": unittest.main()
