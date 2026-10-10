from __future__ import annotations

import unittest

import numpy as np

from modern_classical_benchmark import (
    bounded_weight_contract,
    ess,
    normalise,
    run_contract,
    simulate_truth,
)


class ModernClassicalContractTests(unittest.TestCase):
    def test_bounded_weight_contract_is_normalised_and_capped(self) -> None:
        weights = normalise(np.array([0.995, 0.002, 0.002, 0.001]))
        capped = bounded_weight_contract(weights, cap_factor=2.0)
        self.assertTrue(np.isclose(np.sum(capped), 1.0))
        self.assertLessEqual(np.max(capped), 0.5 + 1e-12)
        self.assertTrue(np.isfinite(ess(capped)))


    def test_all_contracts_return_finite_metrics(self) -> None:
        truth, observations = simulate_truth(np.random.default_rng(7))
        # run_contract reads the evaluation truth through the module-level value;
        # this mirrors run_all and keeps the unit test independent of disk output.
        import modern_classical_benchmark as mcb

        mcb._truth_current = truth
        methods = (
            "bootstrap-systematic",
            "bounded-weight-chopthin",
            "enkf-pf-hybrid",
            "twisted-lookahead",
            "soft-resampling-dpf-forward",
        )
        for i, method in enumerate(methods):
            row = run_contract(method, observations, np.random.default_rng(100 + i), n=30)
            for key in ("rmse", "mae", "mean_ess", "mean_distinct", "runtime_s"):
                self.assertTrue(np.isfinite(row[key]), (method, key, row[key]))
            self.assertEqual(row["likelihood_evals"], 30 * 100)


    def test_fixed_seed_is_numerically_repeatable(self) -> None:
        truth, observations = simulate_truth(np.random.default_rng(11))
        import modern_classical_benchmark as mcb

        mcb._truth_current = truth
        a = run_contract("soft-resampling-dpf-forward", observations, np.random.default_rng(123), n=30)
        mcb._truth_current = truth
        b = run_contract("soft-resampling-dpf-forward", observations, np.random.default_rng(123), n=30)
        for key in ("rmse", "mae", "mean_ess", "mean_distinct", "likelihood_evals"):
            self.assertEqual(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
