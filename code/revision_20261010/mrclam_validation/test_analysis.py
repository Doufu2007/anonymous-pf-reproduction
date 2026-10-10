"""Tests for MR.CLAM result analysis using tiny synthetic *run artifacts*.

These fixtures test the analysis contract only; they are not MR.CLAM data and
must never be reported as external-data results.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_results import (METHODS, _bootstrap_ci, _effect, analyse,
                             compare_replay_dirs, discover, validate_runs)


def write_run(root, dataset, method, seed, pre, post):
    run = root / f"dataset-{dataset:02d}" / "robot-1" / method / f"seed-{seed:02d}"
    run.mkdir(parents=True)
    truth = np.zeros((2, 3))
    # Errors are deliberately different in x/y so radial RMSE is tested.
    rows = [[0., 0., 0., 0., pre, 0., 0., post, 0., 0.],
            [1., 0., 0., 0., 0., pre, 0., 0., post, 0.]]
    with (run / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "truth_x", "truth_y", "truth_theta",
                    "pre_move_x", "pre_move_y", "pre_move_theta",
                    "post_move_x", "post_move_y", "post_move_theta"])
        w.writerows(rows)
    # radial RMSE of rows is |error| for this fixture.
    metrics = {"dataset_id": dataset, "robot_id": 1, "seed": seed,
               "method": method, "rmse_position_radial_m": float(abs(pre)),
               "post_diagnostic_rmse_position_radial_m": float(abs(post)),
               "mean_ess_fraction": .5, "mean_distinct": 10., "runtime_s": 1.,
               "event_count": 2}
    (run / "raw_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    # The manifest's file hashes are intentionally omitted in this fixture;
    # discover/analyse tests do not claim provenance validation.
    (run / "manifest.json").write_text(json.dumps({
        "status": "completed", "dataset_id": dataset, "robot_id": 1,
        "seed": seed, "method": method, "files_sha256": {}}), encoding="utf-8")


class AnalysisTests(unittest.TestCase):
    def test_bootstrap_and_effect_are_deterministic_descriptive(self):
        a = np.array([1., 2., 3.])
        self.assertEqual(_bootstrap_ci(a, seed=1, draws=1000), _bootstrap_ci(a, seed=1, draws=1000))
        self.assertAlmostEqual(_effect(a - 2.), 0.)

    def test_pairing_within_dataset_and_no_p_values(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for dataset in (8, 9):
                for seed in (0, 1):
                    for i, method in enumerate(METHODS):
                        write_run(root, dataset, method, seed,
                                  pre=1. + i + .1 * seed,
                                  post=2. + i + .1 * seed)
            rows = discover(root)
            primary, deltas, secondary, counts = analyse(rows, bootstrap_draws=100)
            self.assertEqual(len(primary), 2 * 4 * 2)
            self.assertEqual(len(deltas), 2 * 3 * 2)
            self.assertEqual(len(secondary), 2 * 4 * 3)
            self.assertEqual(len(counts), 2 * 4)
            self.assertTrue(all("p" not in row for row in deltas))
            # RM-stable - B-stable is exactly +1 m for this fixture.
            rm = next(x for x in deltas if x["dataset_id"] == 8 and
                      x["method"] == "RM-stable" and x["phase"] == "pre_move_radial_rmse_m")
            self.assertAlmostEqual(rm["mean_delta_m"], 1.)

    def test_validation_recomputes_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_run(root, 8, "B-stable", 0, 1., 2.)
            rows = discover(root)
            report = validate_runs(rows, expected=1)
            # The fixture's empty file-hash map is acceptable, and the
            # endpoint is recomputed from predictions.csv.
            self.assertTrue(report["all_manifests_valid"])
            self.assertTrue(report["all_hashes_valid"])
            self.assertTrue(report["all_metrics_recomputed"])

    def test_replay_compares_arrays_and_ignores_timing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); a = root / "a"; b = root / "b"
            for p in (a, b):
                p.mkdir()
                np.save(p / "estimate_pre.npy", np.array([1., 2.]))
                (p / "raw_metrics.json").write_text(json.dumps({"x": 1, "runtime_s": 2.}), encoding="utf-8")
            (b / "raw_metrics.json").write_text(json.dumps({"x": 1, "runtime_s": 999.}), encoding="utf-8")
            report = compare_replay_dirs(a, b)
            self.assertTrue(report["exact"])
            self.assertEqual(report["scientific_arrays_exact"], 1)
            self.assertEqual(report["non_timing_metrics_exact"], 1)


if __name__ == "__main__":
    unittest.main()
