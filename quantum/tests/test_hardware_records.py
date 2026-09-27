"""
Per-window hardware records carry what's needed to match a job to the
vendor's dashboard (job_id, backend, shots). No real hardware: the IQM
backend and job are stand-ins.
"""

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

    counts, timing, job_id, machine = solver._run_iqm_sampler(None, None, 2, 500)

    assert counts == {"01": 3}
    assert job_id == _FakeJob.job_id
    assert machine == "IQM-garnet"
