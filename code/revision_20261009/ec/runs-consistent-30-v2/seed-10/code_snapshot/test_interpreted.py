"""Gold, property, and contract tests for the one declared EC interpretation."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest
import numpy as np

from ecqpf_interpreted import (
    OperatorConfig, consistent_select, duplicate_indices, exact_fitness_subset,
    observe_binary, subset_logfitness, update_swarm_consistent,
)

HISTORICAL = Path(__file__).resolve().parents[2] / 'source_aligned_audit' / 'ecqpf_operator.py'
spec = importlib.util.spec_from_file_location('literal_ec_operator', HISTORICAL)
literal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(literal)


class FixedUniforms:
    """Always observes one nonempty mask, making early-stop counts exact."""
    def __init__(self):
        self.initial = True
    def random(self, shape):
        if self.initial:
            self.initial = False
            return np.full(shape, .5, dtype=np.float64)
        return np.full(shape, .99, dtype=np.float64)


class InterpretedOperatorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = OperatorConfig()

    def test_literal_and_interpreted_direction_gold(self):
        q = np.full((1, 2), .5)
        best = np.array([[0, 1]], dtype=np.int8)
        old_q = literal.update_swarm(q, best, best[0], self.cfg)
        new_q = update_swarm_consistent(q, best, best[0], self.cfg)
        np.testing.assert_allclose(old_q, [[.26, .74]], rtol=0, atol=1e-15)
        np.testing.assert_allclose(new_q, [[.74, .26]], rtol=0, atol=1e-15)
        desired_old = np.where(best == 0, old_q, 1 - old_q)
        desired_new = np.where(best == 0, new_q, 1 - new_q)
        np.testing.assert_allclose(desired_old, [[.26, .26]], rtol=0, atol=1e-15)
        np.testing.assert_allclose(desired_new, [[.74, .74]], rtol=0, atol=1e-15)

    def test_fixed_best_converges_to_alpha(self):
        best = np.array([[0, 1, 0, 1]], dtype=np.int8)
        q = np.array([[0., 1., .37, .89]])
        for _ in range(50):
            q = update_swarm_consistent(q, best, best[0], self.cfg)
        np.testing.assert_allclose(np.where(best == 0, q, 1-q), .8, rtol=0, atol=2e-15)

    def test_qone_representation_is_equivalent_including_ties(self):
        rng = np.random.default_rng(919)
        qzero = rng.random((5, 7)); uzero = rng.random((5, 7))
        uzero[0] = qzero[0]
        np.testing.assert_array_equal(observe_binary(qzero, uzero), ((1-qzero) > (1-uzero)).astype(np.int8))
        pbest = rng.integers(0, 2, size=qzero.shape); gbest = rng.integers(0, 2, size=7)
        qone_next = (self.cfg.c1*(1-qzero) + self.cfg.c2*(self.cfg.alpha*pbest+(1-self.cfg.alpha)*(1-pbest))
                     + (1-self.cfg.c1-self.cfg.c2)*(self.cfg.alpha*gbest+(1-self.cfg.alpha)*(1-gbest)))
        np.testing.assert_allclose(1-update_swarm_consistent(qzero, pbest, gbest, self.cfg), qone_next, rtol=0, atol=3e-16)

    def test_probability_shape_and_coefficient_guards(self):
        q = np.full((2, 3), .5); p = np.zeros((2, 3), dtype=np.int8); g = np.zeros(3, dtype=np.int8)
        for bad_p, bad_g in [(p[:, :2], g), (p, g[:2]), (p, np.zeros((1, 3)))]:
            with self.assertRaises(ValueError):
                update_swarm_consistent(q, bad_p, bad_g, self.cfg)
        for badq in [q*np.nan, q+1, q-1]:
            with self.assertRaises(ValueError):
                update_swarm_consistent(badq, p, g, self.cfg)
        for badcfg in [replace(self.cfg, alpha=1.01), replace(self.cfg, c1=.8, c2=.2), replace(self.cfg, c1=-.1)]:
            with self.assertRaises(ValueError):
                update_swarm_consistent(q, p, g, badcfg)
        with self.assertRaises(ValueError):
            update_swarm_consistent(q, p+2, g, self.cfg)
        with self.assertRaises(ValueError):
            observe_binary(q, np.ones((2, 2)))

    def test_measurement_probability_boundaries(self):
        np.testing.assert_array_equal(observe_binary(np.array([[0., 1., .5]]), np.array([[.1, .99, .5]])), [[1, 0, 0]])
        with self.assertRaises(ValueError):
            observe_binary(np.array([[np.nan]]), np.array([[.5]]))

    def test_nonempty_fitness_is_bounded_by_nearest_singleton(self):
        distances = np.array([.2, 2., 7., 1.]); variance = .7
        masks = np.array([[int((i >> j)&1) for j in range(4)] for i in range(16)])
        scores = subset_logfitness(masks, distances, variance)
        self.assertTrue(np.isneginf(scores[0]))
        self.assertTrue(np.all(scores[1:] <= -.2/(2*variance)+1e-15))
        np.testing.assert_array_equal(exact_fitness_subset(distances), [0])

    def test_duplication_balanced_indices_and_actual_mean(self):
        chosen = duplicate_indices(np.array([1, 3, 5]), 8)
        np.testing.assert_array_equal(chosen, [1, 3, 5, 1, 3, 5, 1, 3])
        particles = np.arange(12).reshape(6, 2)
        np.testing.assert_array_equal(particles[chosen].mean(axis=0), np.array([44., 52.])/8)
        with self.assertRaises(ValueError):
            duplicate_indices([], 8)

    def test_contract_history_mean_ellipse_and_operation_counts(self):
        rng = np.random.default_rng(920)
        cloud = rng.normal(size=(30, 2)); center = np.array([.1, -.2])
        cfg = replace(self.cfg, particle_count=30, swarm_size=30)
        result = consistent_select(cloud, center, cfg, rng)
        np.testing.assert_array_equal(result['ellipse_indices'], literal.ellipse_indices(cloud, center, cfg.ellipse_scale))
        np.testing.assert_allclose(result['estimate'], cloud[result['indices']].mean(axis=0), rtol=0, atol=0)
        self.assertEqual(len(result['indices']), 30)
        self.assertTrue(set(result['indices']).issubset(set(result['ellipse_indices'])))
        history = np.array(result['generation_history'])
        self.assertTrue(np.all(np.diff(history[:, 1]) >= 0))
        self.assertLessEqual(result['logfitness'], result['optimal_logfitness']+1e-12)
        self.assertEqual(result['fitness_evaluations'], cfg.swarm_size*result['generations'])
        self.assertEqual(result['binary_dimensions_processed'], cfg.swarm_size*result['ellipse_count']*result['generations'])
        self.assertEqual(result['operation_counts']['duplication_output_indices'], cfg.particle_count)
        self.assertEqual(result['operation_counts']['rotation_events'], 0)
        self.assertEqual(result['operation_counts']['likelihood_evaluations'], 0)

    def test_unchanged_best_stop_preserves_update_branch_count(self):
        cloud = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        cfg = replace(self.cfg, particle_count=4, swarm_size=3)
        result = consistent_select(cloud, np.zeros(2), cfg, FixedUniforms())
        self.assertEqual(result['generations'], 11)
        self.assertEqual(result['probability_update_steps'], 10)
        self.assertEqual(result['operation_counts']['measurement_binary_entries'], 3*4*11)
        self.assertEqual(result['operation_counts']['probability_update_entries'], 3*4*10)
        self.assertEqual(result['operation_counts']['global_best_replacements'], 1)
        self.assertEqual(result['operation_counts']['personal_best_replacements'], 3)

    def test_same_rng_same_scientific_output(self):
        cloud = np.random.default_rng(921).normal(size=(25, 2))
        cfg = replace(self.cfg, particle_count=25, swarm_size=25)
        first = consistent_select(cloud, np.zeros(2), cfg, np.random.default_rng(922))
        second = consistent_select(cloud, np.zeros(2), cfg, np.random.default_rng(922))
        for key in first:
            if key != 'runtime_s':
                if isinstance(first[key], np.ndarray):
                    np.testing.assert_array_equal(first[key], second[key])
                else:
                    self.assertEqual(first[key], second[key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
