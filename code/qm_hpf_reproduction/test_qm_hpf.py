import unittest
from unittest.mock import patch

import numpy as np
import qm_hpf

from qm_hpf import (
    FilterConfig,
    ModelConfig,
    decode_scalar,
    encode_scalar,
    normalize_logweights,
    quantum_rotation_candidate,
    run_filter,
    simulate_truth,
)


class QMHPFTests(unittest.TestCase):
    def test_fixed_point_round_trip_is_bounded(self):
        cfg = FilterConfig(bit_length=12, state_min=-10.0, state_max=10.0)
        for value in (-100.0, -2.5, 0.0, 3.14, 100.0):
            decoded = decode_scalar(encode_scalar(value, cfg), cfg)
            self.assertGreaterEqual(decoded, cfg.state_min)
            self.assertLessEqual(decoded, cfg.state_max)


    def test_rotation_candidate_is_finite_and_bounded(self):
        cfg = FilterConfig()
        rng = np.random.default_rng(1)
        for generation in range(cfg.generation_count):
            candidate = quantum_rotation_candidate(-2.0, 3.0, 0.1, 0.2, generation, cfg, rng)
            self.assertTrue(np.isfinite(candidate))
            self.assertGreaterEqual(candidate, cfg.state_min)
            self.assertLessEqual(candidate, cfg.state_max)


    def test_weights_are_normalized(self):
        weights = normalize_logweights(np.array([-1000.0, -2.0, -3.0]))
        np.testing.assert_allclose(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights >= 0))

    def test_identity_update_preserves_relative_weights(self):
        """An unchanged particle must keep its original normalized weight."""
        particles = np.array([-2.0, 0.0, 2.0])
        weights = np.array([0.6, 0.3, 0.1])
        cfg = FilterConfig(particle_count=3, generation_count=1)
        counter = {
            "likelihood_evals": 0,
            "candidate_evals": 0,
            "old_evals": 0,
            "proposal_attempts": 0,
            "accepts": 0,
        }
        captured = {}

        def capture_resample(current_weights, rng):
            captured["weights"] = current_weights.copy()
            return np.arange(len(current_weights))

        with patch.object(qm_hpf, "systematic_resample", side_effect=capture_resample):
            qm_hpf._resample_update(
                particles,
                weights,
                lambda low, high, low_weight, high_weight, generation: low,
                np.random.default_rng(123),
                cfg,
                y=0.5,
                k=1,
                obs_variance=1.0,
                counter=counter,
            )

        np.testing.assert_allclose(captured["weights"], weights)


    def test_simulation_and_filter_are_reproducible(self):
        model = ModelConfig(horizon=20)
        cfg = FilterConfig(particle_count=30, generation_count=3, bit_length=10)
        truth_a, obs_a = simulate_truth(np.random.default_rng(7), model)
        truth_b, obs_b = simulate_truth(np.random.default_rng(7), model)
        np.testing.assert_array_equal(truth_a, truth_b)
        np.testing.assert_array_equal(obs_a, obs_b)
        a = run_filter(obs_a, truth_a, np.random.default_rng(11), "qm_hpf", model, cfg)
        b = run_filter(obs_b, truth_b, np.random.default_rng(11), "qm_hpf", model, cfg)
        np.testing.assert_array_equal(a["estimate"], b["estimate"])
        self.assertTrue(np.isfinite(a["rmse"]))


if __name__ == "__main__":
    unittest.main()
