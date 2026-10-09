import itertools
import unittest
import numpy as np
from ecqpf_operator import (
    OperatorConfig, ellipse_indices, subset_logfitness, observe_binary,
    update_swarm, duplicate_indices, exact_fitness_subset, source_select,
    shared_classical_select,
    full_systematic_select,
)

class SourceOperatorTests(unittest.TestCase):
    def test_classical_uniform_likelihood_gold(self):
        p=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        result=full_systematic_select(p,np.zeros(2),OperatorConfig(particle_count=4),np.random.default_rng(5),1.)
        np.testing.assert_array_equal(result['indices'],np.arange(4))
        np.testing.assert_allclose(result['estimate'],[0.,0.],rtol=0,atol=1e-14)
        self.assertAlmostEqual(result['sampling_ess'],4.)
        self.assertEqual(result['likelihood_evaluations'],4)

    def test_ellipse_gold_and_rigid_motion(self):
        p = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        np.testing.assert_array_equal(ellipse_indices(p, np.zeros(2), 2.), np.arange(4))
        np.testing.assert_array_equal(ellipse_indices(p, np.zeros(2), .5), np.array([],dtype=int))
        r = np.array([[.6,-.8],[.8,.6]])
        shift = np.array([7.,-3.])
        np.testing.assert_array_equal(ellipse_indices(p@r.T+shift,shift,2.01),np.arange(4))

    def test_fitness_hand_calculation_and_empty(self):
        bits = np.array([[1,0],[1,1],[0,0]])
        answer = subset_logfitness(bits,np.array([1.,3.]),2.)
        np.testing.assert_allclose(answer[:2],[-.25,-.5],rtol=0,atol=1e-14)
        self.assertTrue(np.isneginf(answer[2]))

    def test_literal_eq12_observation(self):
        q = np.array([[0.,.2,.8,1.]])
        u = np.array([[.1,.1,.9,.9]])
        np.testing.assert_array_equal(observe_binary(q,u),[[1,0,1,0]])

    def test_eq23_to25_hand_calculation(self):
        cfg=OperatorConfig()
        out=update_swarm(np.array([[.3,.7]]),np.array([[1,0]]),np.array([1,0]),cfg)
        np.testing.assert_allclose(out,[[.7,.3]],atol=1e-14,rtol=0)
        rng=np.random.default_rng(81)
        q=rng.random((8,17)); b=rng.integers(0,2,(8,17)); g=rng.integers(0,2,17)
        out=update_swarm(q,b,g,cfg)
        self.assertTrue(np.all((out>=0)&(out<=1)))

    def test_exact_classical_optimizer_global_bound(self):
        distances=np.array([3.,1.,2.,4.])
        selected=exact_fitness_subset(distances)
        np.testing.assert_array_equal(selected,[1])
        all_bits=np.array([bits for bits in itertools.product([0,1],repeat=4)])
        opt=subset_logfitness(np.array([[0,1,0,0]]),distances,2.)[0]
        self.assertTrue(np.all(subset_logfitness(all_bits,distances,2.)<=opt+1e-14))

    def test_duplication_balanced_and_estimate(self):
        indices=duplicate_indices(np.array([2,5,7]),8)
        np.testing.assert_array_equal(indices,[2,5,7,2,5,7,2,5])
        counts=np.bincount(indices)[[2,5,7]]
        self.assertLessEqual(np.ptp(counts),1)

    def test_source_module_valid_identity_and_repeat(self):
        cfg=OperatorConfig(particle_count=30,swarm_size=12,iteration_max=6)
        particles=np.random.default_rng(71).normal(size=(30,2))
        first=source_select(particles,np.array([.2,.3]),cfg,np.random.default_rng(15))
        second=source_select(particles,np.array([.2,.3]),cfg,np.random.default_rng(15))
        np.testing.assert_array_equal(first['indices'],second['indices'])
        self.assertEqual(len(first['indices']),30)
        self.assertTrue(set(first['indices']).issubset(set(first['ellipse_indices'])))
        self.assertLessEqual(first['logfitness'],first['optimal_logfitness']+1e-12)
        self.assertIsNone(first['sampling_ess'])
        np.testing.assert_allclose(first['estimate'],particles[first['indices']].mean(axis=0),atol=1e-14,rtol=0)

    def test_shared_constraint_and_empty_guardrail(self):
        cfg=OperatorConfig(particle_count=30,swarm_size=12,iteration_max=6)
        particles=np.random.default_rng(12).normal(size=(30,2)); center=np.array([.1,.2])
        source=source_select(particles,center,cfg,np.random.default_rng(13))
        for method in ['ellipse_exact_fitness','ellipse_systematic_fitness']:
            classical=shared_classical_select(particles,center,cfg,np.random.default_rng(14),method)
            np.testing.assert_array_equal(source['ellipse_indices'],classical['ellipse_indices'])
            self.assertTrue(set(classical['indices']).issubset(set(source['ellipse_indices'])))
        with self.assertRaises(ValueError):
            source_select(particles,np.array([1e5,1e5]),cfg,np.random.default_rng(16))

if __name__=='__main__':
    unittest.main()
