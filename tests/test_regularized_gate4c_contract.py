from __future__ import annotations

import unittest

from types import SimpleNamespace

from scripts.fathi_benchmark.regularization.regularized_pipeline_artifacts import (
    BOOTSTRAP_PARENT_CLASSIFICATION,
    compose_frozen_regularized_objective,
    require_regularized_parent_identity,
)


class Gate4CRegularizedObjectiveTests(unittest.TestCase):
    def test_frozen_regularized_objective_components(self):
        value = compose_frozen_regularized_objective(
            data_objective=10.0,
            q_lambda=2.0,
            q_mu=4.0,
            beta_lambda=0.5,
            beta_mu=0.25,
        )
        self.assertEqual(value["J_reg_lambda"], 1.0)
        self.assertEqual(value["J_reg_mu"], 1.0)
        self.assertEqual(value["J_reg"], 2.0)
        self.assertEqual(value["J_total"], 12.0)

    def test_beta_is_not_divided_by_two(self):
        value = compose_frozen_regularized_objective(
            data_objective=0.0,
            q_lambda=8.0,
            q_mu=0.0,
            beta_lambda=0.5,
            beta_mu=1.0,
        )
        self.assertEqual(value["J_reg_lambda"], 4.0)
        self.assertNotEqual(value["J_reg_lambda"], 2.0)

    def test_invalid_negative_tv_blocked(self):
        with self.assertRaises(ValueError):
            compose_frozen_regularized_objective(
                data_objective=1.0,
                q_lambda=-1.0,
                q_mu=1.0,
                beta_lambda=0.5,
                beta_mu=0.5,
            )


    def _paths(self, parent: int = 0):
        return SimpleNamespace(
            identity=SimpleNamespace(
                run_id="fathi_s43_repro_tv_p20_t052",
                parent_iteration=parent,
                child_iteration=parent + 1,
                transition_id=f"iter_{parent:03d}_to_iter_{parent + 1:03d}",
            )
        )

    def _bootstrap_parent(self):
        return {
            "run": "fathi_s43_repro_tv_p20_t052",
            "run_id": "fathi_s43_repro_tv_p20_t052",
            "iter": 0,
            "lineage": {
                "classification": BOOTSTRAP_PARENT_CLASSIFICATION,
                "optimizer_history_reused": False,
            },
            "material_sha256": {"Mat_0_Kappa.h5": "abc"},
        }

    def test_iter000_bootstrap_parent_identity_is_accepted(self):
        mode = require_regularized_parent_identity(
            self._bootstrap_parent(),
            self._paths(0),
            label="parent",
        )
        self.assertEqual(mode, "CERTIFIED_ITER000_BOOTSTRAP_IDENTITY")

    def test_iter000_bootstrap_wrong_run_is_blocked(self):
        parent = self._bootstrap_parent()
        parent["run_id"] = "wrong-run"
        with self.assertRaisesRegex(ValueError, "bootstrap run_id mismatch"):
            require_regularized_parent_identity(
                parent,
                self._paths(0),
                label="parent",
            )

    def test_iter000_bootstrap_partial_transition_identity_is_blocked(self):
        parent = self._bootstrap_parent()
        parent["parent_iteration"] = 0
        with self.assertRaisesRegex(ValueError, "partial transition identity"):
            require_regularized_parent_identity(
                parent,
                self._paths(0),
                label="parent",
            )

    def test_later_parent_without_transition_identity_is_blocked(self):
        parent = self._bootstrap_parent()
        parent["iter"] = 1
        with self.assertRaisesRegex(ValueError, "outside iter000 bootstrap"):
            require_regularized_parent_identity(
                parent,
                self._paths(1),
                label="parent",
            )

    def test_full_transition_identity_keeps_existing_contract(self):
        manifest = {
            "run_id": "fathi_s43_repro_tv_p20_t052",
            "parent_iteration": 1,
            "child_iteration": 2,
            "transition": "iter_001_to_iter_002",
        }
        mode = require_regularized_parent_identity(
            manifest,
            self._paths(1),
            label="parent",
        )
        self.assertEqual(mode, "TRANSITION_IDENTITY")


if __name__ == "__main__":
    unittest.main()
