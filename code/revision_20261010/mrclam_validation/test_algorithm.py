"""Gold, wrap-boundary, posterior and operation-count tests; no real data."""
import unittest
import numpy as np

from algorithm import (Config, COUNTER_NAMES, METHODS, advance_unicycle,
                       decode_states, encode_states,
                       log_likelihood, normalize_logweights, run_filter,
                       weighted_pose_mean, wrap_angle, accept_independence_prior_move)


class ModelTests(unittest.TestCase):
    def test_unicycle_exact_straight_and_arc(self):
        states = np.array([[0., 0., 0.], [0., 0., np.pi / 2]])
        straight = advance_unicycle(states, np.array([[2., 1., 0.]]))
        np.testing.assert_allclose(straight[:, :2], [[2., 0.], [0., 2.]], atol=1e-12)
        curved = advance_unicycle(states[:1], np.array([[np.pi / 2, 1., 1.]]))
        np.testing.assert_allclose(curved, [[1., 1., np.pi / 2]], atol=1e-12)

    def test_seam_likelihood_and_circular_mean(self):
        cfg = Config()
        states = np.array([[0., 0., np.pi - .01], [0., 0., -np.pi + .01]])
        ll = log_likelihood(states, np.array([[1., 0.]]), np.array([[1., np.pi]]), cfg)
        np.testing.assert_allclose(ll[0], ll[1], atol=1e-12)
        mean = weighted_pose_mean(states, np.array([.5, .5]))
        self.assertAlmostEqual(abs(mean[2]), np.pi, places=12)

    def test_log_weights_underflow_stability(self):
        weights = normalize_logweights(np.array([-100000., -100001.]))
        np.testing.assert_allclose(weights, [1. / (1. + np.exp(-1.)),
                                           1. / (1. + np.exp(1.))])

    def test_prior_independence_mh_cancellation(self):
        self.assertTrue(accept_independence_prior_move(-2., -1., .99))
        self.assertFalse(accept_independence_prior_move(-1., -2., .99))
        self.assertTrue(accept_independence_prior_move(-1., -2., .1))

    def test_likelihood_observation_permutation(self):
        cfg = Config()
        states = np.array([[1., 2., .5], [.5, -.5, -3.]])
        xy = np.array([[2., 2.], [-1., 0.]])
        obs = np.array([[1.1, -.4], [2.5, -2.2]])
        np.testing.assert_allclose(log_likelihood(states, xy, obs, cfg),
                                   log_likelihood(states, xy[::-1], obs[::-1], cfg))

    def test_map_encoding_boundaries_and_translation(self):
        cfg = Config(state_min=(-2., -4., -np.pi), state_max=(2., 4., np.pi))
        states = np.array([[-2., -4., -np.pi], [2., 4., np.pi]])
        reconstructed = decode_states(encode_states(states, cfg), cfg)
        np.testing.assert_allclose(reconstructed[:, :2], states[:, :2])
        np.testing.assert_allclose(wrap_angle(reconstructed[:, 2] - states[:, 2]), 0., atol=1e-12)
        shifts = np.array([8., -3.])
        landmarks = np.array([[1., 1.], [2., -1.]])
        observations = np.array([[2., 1.], [3., 2.]])
        moved = states.copy()
        moved[:, :2] += shifts
        np.testing.assert_allclose(log_likelihood(states, landmarks, observations, cfg),
                                   log_likelihood(moved, landmarks + shifts, observations, cfg))

    def test_initial_center_is_ignored(self):
        segments = [np.array([[.1, .2, 0.]])]
        landmarks = [np.array([[1., 1.]])]
        observations = [np.array([[np.sqrt(2.), np.pi / 4]])]
        cfg = Config(budget=120, quantum_particles=20)
        a = run_filter(segments, landmarks, observations, [0., 0., 0.], 'B-stable', 14, cfg)
        b = run_filter(segments, landmarks, observations, [1000., 1000., 3.], 'B-stable', 14, cfg)
        np.testing.assert_array_equal(a['initial_particles'], b['initial_particles'])
        np.testing.assert_array_equal(a['estimate_pre'], b['estimate_pre'])

    def test_synthetic_event_count_and_seed_replay(self):
        segments = [np.array([[.1, .2, 0.]]) for _ in range(5)]
        landmarks = [np.array([[1., 1.]]) for _ in range(5)]
        observations = [np.array([[np.sqrt(2.), np.pi / 4]]) for _ in range(5)]
        cfg = Config(budget=120, quantum_particles=20)
        for method in METHODS:
            a = run_filter(segments, landmarks, observations, [0., 0., 0.], method, 14, cfg)
            b = run_filter(segments, landmarks, observations, [0., 0., 0.], method, 14, cfg)
            np.testing.assert_array_equal(a['estimate_pre'], b['estimate_pre'])
            np.testing.assert_array_equal(a['estimate_post'], b['estimate_post'])
            self.assertEqual(a['counters']['likelihood_evals'], 5 * cfg.budget)
            self.assertTrue(all(a['guardrails'].values()))
            self.assertTrue(np.all(np.abs(a['estimate_pre'][:, 2]) <= np.pi))
            self.assertEqual(a['step_counters'].shape, (5, len(COUNTER_NAMES)))


if __name__ == '__main__':
    unittest.main()
