from __future__ import annotations

from types import SimpleNamespace

from scripts.fathi_benchmark.run_certified_external_exact_reverse import (
    runtime_objective_dt,
    runtime_objective_dt_source,
)


def test_runtime_objective_dt_prefers_certified_quadrature_contract():
    runtime = {
        "objective_dt": 0.0005,
        "driver": SimpleNamespace(dt=0.0004999999999999999),
    }
    assert runtime_objective_dt(runtime) == 0.0005
    assert runtime_objective_dt_source(runtime) == "certified_reference.contract.dt"


def test_runtime_objective_dt_falls_back_to_driver_for_legacy_route():
    runtime = {
        "driver": SimpleNamespace(dt=0.0005),
    }
    assert runtime_objective_dt(runtime) == 0.0005
    assert runtime_objective_dt_source(runtime) == "driver.dt"
