from __future__ import annotations

import unittest

import numpy as np

from scripts.fathi_benchmark.certified_data_objective import (
    certified_data_objective,
)


class CertifiedDataObjectiveContractTests(unittest.TestCase):
    def test_certified_dt_is_authoritative_for_quadrature(self):
        contract_dt = 0.0005
        driver_dt = float(np.nextafter(contract_dt, 0.0))
        self.assertNotEqual(driver_dt, contract_dt)

        current = np.arange(24, dtype=np.float64).reshape(4, 2, 3) / 7.0
        truth = np.zeros_like(current)

        certified = certified_data_objective(
            current,
            truth,
            certified_dt=contract_dt,
            driver_dt=driver_dt,
        )
        legacy = certified_data_objective(
            current,
            truth,
            certified_dt=driver_dt,
            driver_dt=driver_dt,
        )

        self.assertEqual(certified.objective_dt, contract_dt)
        self.assertEqual(certified.driver_dt, driver_dt)
        self.assertEqual(certified.dt_ulp_distance, 1)
        self.assertNotEqual(certified.value, legacy.value)

    def test_driver_dt_outside_contract_tolerance_is_blocked(self):
        current = np.ones((4, 1, 1), dtype=np.float64)
        truth = np.zeros_like(current)
        with self.assertRaises(ValueError):
            certified_data_objective(
                current,
                truth,
                certified_dt=0.0005,
                driver_dt=0.0005000001,
            )


if __name__ == "__main__":
    unittest.main()
