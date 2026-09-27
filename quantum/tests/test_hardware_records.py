"""
Per-window hardware records carry what's needed to match a job to the
vendor's dashboard (job_id, backend, shots). No real hardware: the IQM
backend and job are stand-ins.
"""

import pytest

from quantum.solvers.Pennylane_solver import PennylaneSolver
import quantum.hardware.iqm_backend as iqm_backend


class _FakeJob:
    job_id = "01a0e468-0000-0000-0000-000000000000"

    def result(self):
        return self

    def get_counts(self):
        return {"01": 3}


class _FakeQrispBackend:
    name = "IQM-garnet"

    def run_async(self, circuit, shots):
        return _FakeJob()


def test_iqm_sampler_returns_job_id_and_machine(monkeypatch):
    solver = PennylaneSolver.__new__(PennylaneSolver)
    monkeypatch.setattr(solver, "_pennylane_tape_to_qrisp", lambda *a: "circuit", raising=False)
    monkeypatch.setattr(solver, "_get_backend", lambda n: _FakeQrispBackend(), raising=False)
    monkeypatch.setattr(
        iqm_backend.IQMHardwareBackend, "_record_execution", lambda self, job, shots: None
    )

    counts, telemetry = solver._run_iqm_sampler(None, None, 2, 500)

    assert counts == {"01": 3}
    assert telemetry["job_id"] == _FakeJob.job_id
    assert telemetry["backend"] == "IQM-garnet"
    assert set(telemetry) == {"iqm_timing", "job_id", "backend", "estimated_credits"}


@pytest.mark.parametrize(
    "machine, execution_sec, compile_sec, charged",
    [
        # Resonance dashboard charges, 2026-09-27
        ("IQM-garnet", 1.556093, 0.089856, 1.0),
        ("IQM-garnet", 0.474861, 0.088483, 0.5),
        ("IQM-emerald", 0.622382, 0.164862, 0.75),
        ("emerald", 0.734884, 0.168167, 0.75),
        # sub-second execution, but slow compilation makes it a 2 s job
        ("IQM-emerald", 0.53, 0.496, 1.5),
    ],
)
def test_iqm_credit_estimate_matches_observed_charges(
    machine, execution_sec, compile_sec, charged
):
    assert iqm_backend.estimate_iqm_credits(machine, execution_sec, compile_sec) == charged


def test_iqm_credit_estimate_unknown_machine_or_timing():
    assert iqm_backend.estimate_iqm_credits("IQM-sirius", 0.5) is None
    assert iqm_backend.estimate_iqm_credits("IQM-garnet", None) is None
