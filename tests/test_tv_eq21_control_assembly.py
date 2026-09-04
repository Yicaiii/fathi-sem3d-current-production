import numpy as np

from scripts.fathi_benchmark.regularization.tv_weight import fathi_eq24_weight


def test_mpa_to_pa_covector_chain_rule() -> None:
    q_mpa = np.array([2.0, -3.0, 5.0], dtype=np.float64)
    dm_pa = np.array([7.0e4, -2.0e4, 1.5e5], dtype=np.float64)
    q_pa = q_mpa / 1.0e6
    dm_mpa = dm_pa / 1.0e6
    assert np.isclose(q_mpa @ dm_mpa, q_pa @ dm_pa, rtol=0.0, atol=1.0e-15)


def test_eq24_weighted_regularization_has_varrho_data_norm() -> None:
    gmis = np.array([2.0, -1.0, 4.0, 3.0], dtype=np.float64)
    greg = np.array([1.0e-6, 3.0e-6, -2.0e-6, 5.0e-6], dtype=np.float64)
    varrho = 0.5
    weight = fathi_eq24_weight(gmis, greg, varrho=varrho)
    ratio = np.linalg.norm(weight.weight * greg) / np.linalg.norm(gmis)
    assert np.isclose(ratio, varrho, rtol=1.0e-14, atol=1.0e-14)
