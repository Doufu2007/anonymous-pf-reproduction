"""Gold, mathematical and accounting tests for the isolated budget audit."""
from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from budget_benchmark import (conditional_log_ratio, resample_pairs,
                              run_budget_filter, qm_budget_update)
from qm_hpf import FilterConfig, ModelConfig, normalize_logweights, particle_logweights, simulate_truth, transition


class BudgetBenchmarkTests(unittest.TestCase):
    def test_full_mh_density_ratio_gold_and_reverse(self):
        # Set the ancestor so its transition mean is zero. At k>30 the
        # observation is h(x)=0.5x-2. y=-2 makes its mean zero at x=0.
        k = 31
        ancestor = -2.0 * (1.0 + np.sin(0.04 * np.pi * k))
        model = ModelConfig()
        ratio = conditional_log_ratio(0.0, 2.0, ancestor, -2.0, k, model)
        # Transition log ratio = -2; likelihood log ratio = -0.5.
        self.assertAlmostEqual(ratio, -2.5, places=12)
        reverse = conditional_log_ratio(2.0, 0.0, ancestor, -2.0, k, model)
        self.assertAlmostEqual(reverse, 2.5, places=12)
        # min(1,r) / min(1,1/r) equals the target density ratio.
        self.assertAlmostEqual(np.exp(min(0.0, ratio) - min(0.0, reverse)), np.exp(-2.5), places=12)

    def test_mh_target_agrees_with_analytic_conditional_gaussian(self):
        k, ancestor, y, q, r = 45, 0.3, 1.4, 1.0, 1.0
        mu = float(transition(ancestor, k))
        variance = 1.0 / (1.0 / q + 0.25 / r)
        mean = variance * (mu / q + 0.5 * (y + 2.0) / r)
        old, new = -0.4, 2.8
        expected = -((new - mean) ** 2 - (old - mean) ** 2) / (2.0 * variance)
        self.assertAlmostEqual(conditional_log_ratio(old, new, ancestor, y, k, ModelConfig()), expected, places=12)

    def test_resampling_preserves_ancestor_state_and_likelihood_pairs(self):
        ancestors = np.array([11.0, 22.0, 33.0, 44.0])
        particles = ancestors + 0.5
        caches = ancestors * -0.1
        a, x, ll = resample_pairs(ancestors, particles, caches, np.array([0.05, 0.15, 0.2, 0.6]), np.random.default_rng(8))
        assert_allclose(x, a + 0.5, rtol=0, atol=0)
        assert_allclose(ll, a * -0.1, rtol=0, atol=0)

    def test_qm_low_set_does_not_force_budget_or_exceed_generation_cap(self):
        # ESS is close to N, the strict threshold produces exactly one low
        # particle. With G=2 it may only evaluate two candidates, not N.
        cfg = replace(FilterConfig(), particle_count=4, generation_count=2)
        particles = np.array([0.0, 0.1, 0.2, 0.3])
        weights = np.array([0.26, 0.25, 0.25, 0.24])
        cache = particle_logweights(particles, 0.0, 31, 1.0)
        counter = {"likelihood_evals": 4, "proposal_attempts": 0, "candidate_evals": 0,
                   "accepts": 0, "rotation_proposals": 0, "crossover_proposals": 0}
        _, new_weights, attempts = qm_budget_update(particles, weights, cache, 0.0, 31,
                                                     np.random.default_rng(5), ModelConfig(), cfg, counter, 4)
        self.assertEqual(attempts, 2)
        self.assertEqual(counter["likelihood_evals"], 6)
        self.assertAlmostEqual(float(new_weights.sum()), 1.0)
        self.assertTrue(np.all(new_weights == 0.25))

    def test_step_budget_count_normalization_finite_metrics(self):
        model = replace(ModelConfig(), horizon=8)
        truth, obs = simulate_truth(np.random.default_rng(9), model)
        for method in ("bootstrap_200", "resample_move_100", "qm_hpf_like_100"):
            result = run_budget_filter(obs, truth, np.random.default_rng(29), method, model)
            calls = np.asarray(result["step_likelihood_evals"])
            self.assertTrue(np.all(calls <= 200))
            self.assertEqual(result["likelihood_evals"], int(calls.sum()))
            if method != "qm_hpf_like_100":
                assert_array_equal(calls, np.full(model.horizon, 200))
            self.assertTrue(np.isfinite(result["rmse"]))
            self.assertTrue(np.isfinite(result["estimate"]).all())
            self.assertAlmostEqual(result["final_weight_sum"], 1.0)

    def test_fixed_seed_same_environment_reproducibility(self):
        model = replace(ModelConfig(), horizon=5)
        truth, obs = simulate_truth(np.random.default_rng(1), model)
        for method in ("bootstrap_200", "resample_move_100", "qm_hpf_like_100"):
            one = run_budget_filter(obs, truth, np.random.default_rng(2), method, model)
            two = run_budget_filter(obs, truth, np.random.default_rng(2), method, model)
            for key in one:
                if key in {"runtime_s", "ess_per_second"}:
                    continue
                if isinstance(one[key], np.ndarray):
                    assert_array_equal(one[key], two[key])
                else:
                    self.assertEqual(one[key], two[key], key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
