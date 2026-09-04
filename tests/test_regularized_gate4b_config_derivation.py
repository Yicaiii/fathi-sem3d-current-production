from __future__ import annotations

import unittest

from scripts.fathi_benchmark.regularization.derive_regularized_configs import (
    derive_engine,
    derive_runtime,
)


class RegularizedConfigDerivationTests(unittest.TestCase):
    def setUp(self):
        self.base_run = "baseline"
        self.new_run = "regularized"
        self.engine = {
            "run_id": self.base_run,
            "namespace": {
                "data_run_pattern": "data/reproduction/{run_id}",
                "results_run_pattern": "results/{run_id}",
            },
            "optimizer": {
                "fixed_reproduction_scaling": {
                    "J_ref": 3.0,
                    "J_ref_iteration": 0,
                }
            },
            "external_armijo": {
                "c1": 1e-4,
                "rho": 0.5,
                "alpha0": 1.0,
                "maximum_backtracks": 12,
                "objective": "data",
            },
        }
        self.runtime = {
            "benchmark_name": self.base_run,
            "runtime_layout": {
                "initial_parent_workspace": f"data/reproduction/{self.base_run}/iterations/{{parent_tag}}/accepted",
                "iteration_pattern": f"data/reproduction/{self.base_run}/iterations/iter_{{iteration:03d}}",
                "state_pattern": f"results/{self.base_run}/states/iter_{{iteration:03d}}_state.npz",
                "transition_result_pattern": f"results/{self.base_run}/{{transition}}",
                "true_observed_workspace": f"data/reproduction/{self.base_run}/forward/true_layered",
            },
            "production_objective": {
                "classification": "DATA",
                "J0": 2.0,
            },
            "objective": {"total": "old"},
            "new_only_runtime_contract": {
                "active_run_namespace": self.base_run,
                "J0": 2.0,
            },
        }

    def test_engine_namespace_remains_run_driven(self):
        out = derive_engine(
            self.engine,
            new_run=self.new_run,
            regularization_config="configs/reg.json",
        )
        self.assertEqual(out["run_id"], self.new_run)
        self.assertIn("{run_id}", out["namespace"]["data_run_pattern"])
        self.assertTrue(
            out["regularization_contract"]["weights_frozen_within_parent_line_search"]
        )

    def test_runtime_mutable_routes_move_but_true_workspace_stays_certified(self):
        out = derive_runtime(
            self.runtime,
            baseline_run=self.base_run,
            new_run=self.new_run,
            certified_data_j0=3.0,
            regularization_config="configs/reg.json",
        )
        self.assertEqual(out["benchmark_name"], self.new_run)
        layout = out["runtime_layout"]
        self.assertIn(self.new_run, layout["initial_parent_workspace"])
        self.assertIn(self.new_run, layout["iteration_pattern"])
        self.assertIn(self.new_run, layout["state_pattern"])
        self.assertIn(self.new_run, layout["transition_result_pattern"])
        self.assertIn(self.base_run, layout["true_observed_workspace"])
        self.assertEqual(out["production_objective"]["J0"], 3.0)
        self.assertEqual(
            out["production_objective"]["baseline_runtime_J0_before_derivation"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
