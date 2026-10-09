"""Gold, property and metamorphic checks before the revision science batch.

Stdlib unittest and fixed-seed generated properties are used because pytest and
Hypothesis are unavailable in the recorded runtime. No quality-score assertion.
"""
import importlib.util
import math
from pathlib import Path
import unittest
import numpy as np

from qgpf_stable import (ModelConfig, FilterConfig, log_likelihood,
                        normalize_logweights, accept_log_candidate,
                        rotate_probability, run_filter)

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'qgpf_reproduction' / 'qgpf.py'
spec = importlib.util.spec_from_file_location('unchanged_qgpf_for_tests', OLD)
legacy = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)

GOLD = [
    (-1583.489614168062, -31697.69094778979, False),
    (-31697.69094778979, -315135.2680179711, False),
    (-315135.2680179711, -790.8486797922976, True),
    (-790.8486797922976, -635181.7226639752, False),
]

class StableGold(unittest.TestCase):
    def test_four_real_underflow_examples(self):
        for old, new, desired in GOLD:
            self.assertEqual(math.exp(old), 0.0)
            self.assertEqual(math.exp(new), 0.0)
            self.assertTrue(math.exp(new) >= math.exp(old))
            self.assertEqual(accept_log_candidate(old, new, .5, 'ratio'), desired)

    def test_density_hand_computable(self):
        p = np.array([[0.,0.,0.,0.],[3.,0.,4.,0.]])
        result = log_likelihood(p, 5., ModelConfig())
        np.testing.assert_allclose(result, [-12.5-.5*math.log(2*math.pi),-.5*math.log(2*math.pi)], rtol=0,atol=1e-14)

    def test_safe_range_acceptance_matches_ordinary(self):
        for old,new in [(-3.,-1.),(-1.,-3.),(-3.,-3.),(-.1,-.9)]:
            for u in [.01,.25,.5,.9]:
                ordinary = math.exp(new)>=math.exp(old) or u<math.exp(new-old)
                self.assertEqual(accept_log_candidate(old,new,u,'ratio'),ordinary)

    def test_nonfinite_and_outside_support(self):
        self.assertFalse(accept_log_candidate(0.,-math.inf,.5,'ratio'))
        self.assertTrue(accept_log_candidate(-math.inf,0.,.5,'ratio'))
        for old,new in [(-math.inf,-math.inf),(0.,math.nan),(math.inf,0.)]:
            with self.assertRaises(FloatingPointError):accept_log_candidate(old,new,.5,'ratio')
        with self.assertRaises(FloatingPointError):normalize_logweights(np.array([-math.inf,-math.inf]))
        with self.assertRaises(FloatingPointError):log_likelihood(np.array([[math.nan,0.,0.,0.]]),0.,ModelConfig())

    def test_retention_is_strict_improvement_only(self):
        for old,new in [(-100.,-101.),(-100.,-100.),(-100.,-99.)]:
            for u in [.01,.99]:
                self.assertEqual(accept_log_candidate(old,new,u,'retain'),new>old)

class StableProperties(unittest.TestCase):
    def test_normalization_common_offset_and_permutation(self):
        rng=np.random.default_rng(61009)
        for _ in range(100):
            values=rng.normal(0,10,20)
            w=normalize_logweights(values)
            np.testing.assert_allclose(w.sum(),1.,rtol=0,atol=2e-15)
            self.assertTrue(np.isfinite(w).all())
            self.assertTrue((w>=0).all())
            np.testing.assert_allclose(w,normalize_logweights(values+1000.),rtol=0,atol=1e-13)
            perm=rng.permutation(len(values))
            np.testing.assert_allclose(w[perm],normalize_logweights(values[perm]),rtol=0,atol=2e-15)

    def test_extreme_finite_density_has_no_probability_floor(self):
        w=normalize_logweights(np.array([-10000.,-10001.,-10002.]))
        expected=np.exp(np.array([0.,-1.,-2.]));expected/=expected.sum()
        np.testing.assert_allclose(w,expected,rtol=0,atol=1e-15)
        self.assertGreater(w[0],w[1]);self.assertGreater(w[1],w[2])

    def test_scalar_rotation_matches_unchanged_module_exactly(self):
        for p in np.linspace(0.,1.,101):
            for bit in [0,1]:
                for angle in [0.,.01*math.pi,.08*math.pi]:
                    np.testing.assert_allclose(rotate_probability(float(p),bit,angle),legacy._rotate_probability(float(p),bit,angle),rtol=0,atol=0)

    def test_density_agrees_with_ordinary_in_safe_range(self):
        rng=np.random.default_rng(5005);p=rng.normal(0,1,(40,4))
        np.testing.assert_allclose(np.exp(log_likelihood(p,2.,ModelConfig())),legacy.likelihood(p,2.,legacy.ModelConfig()),rtol=2e-15,atol=1e-16)

class MinimalRunProperties(unittest.TestCase):
    def config(self):
        from dataclasses import replace
        return replace(ModelConfig(),horizon=2),replace(FilterConfig(),particle_count=10,bit_length=3,generation_count=2)

    def test_counts_and_same_seed_repro(self):
        model,cfg=self.config();obs=np.array([0.,75.,80.])
        for method in ['B-stable','Q-stable','Q-stable-retain']:
            a=run_filter(obs,np.random.default_rng(44),method,model,cfg)
            b=run_filter(obs,np.random.default_rng(44),method,model,cfg)
            for key in ['estimate_pre','estimate_post','final_particles','final_weights','step_counters']:
                np.testing.assert_allclose(a[key],b[key],rtol=0,atol=0)
            self.assertTrue(a['guardrails']['finite_estimates'])
            self.assertEqual(a['counters']['crossover_operations'],0)
            if method=='B-stable':
                self.assertEqual(a['counters']['likelihood_evals'],20)
                self.assertEqual(a['counters']['candidate_evals'],0)
            else:
                # N=10, elite=2, G=2, T=2.
                self.assertEqual(a['counters']['likelihood_evals'],2*(10+8*(1+2)+10))
                self.assertEqual(a['counters']['candidate_evals'],32)
                self.assertEqual(a['counters']['fitness_evals'],32)
                self.assertEqual(a['counters']['proposal_chromosome_events'],16)
                self.assertEqual(a['counters']['proposal_candidate_attempts'],32)
                self.assertEqual(a['counters']['measurement_bit_observations'],32*4*3*2)
                self.assertEqual(a['counters']['rotation_calls'],a['counters']['probability_entry_updates'])
            self.assertEqual(int(a['step_counters'][:,0].sum()),a['counters']['likelihood_evals'])

    def test_post_endpoint_recording_has_no_state_feedback_or_calls(self):
        model,cfg=self.config();obs=np.array([0.,75.,80.])
        for method in ['B-stable','Q-stable','Q-stable-retain']:
            a=run_filter(obs,np.random.default_rng(77),method,model,cfg,record_post=True)
            b=run_filter(obs,np.random.default_rng(77),method,model,cfg,record_post=False)
            for key in ['estimate_pre','final_particles','final_weights','step_counters']:
                np.testing.assert_allclose(a[key],b[key],rtol=0,atol=0)
            self.assertEqual(a['counters'],b['counters'])
            self.assertEqual(a['rng_final_state'],b['rng_final_state'])

if __name__=='__main__':unittest.main(verbosity=2)
