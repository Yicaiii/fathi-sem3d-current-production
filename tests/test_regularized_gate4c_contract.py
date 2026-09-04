from __future__ import annotations

import unittest

from scripts.fathi_benchmark.regularization.regularized_pipeline_artifacts import (
    compose_frozen_regularized_objective,
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


if __name__ == "__main__":
    unittest.main()
