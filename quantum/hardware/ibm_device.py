"""
Custom PennyLane device for IBM Quantum hardware (device="qiskit.remote"),
adding job telemetry that the stock pennylane_qiskit.QiskitDevice discards.

pennylane_qiskit's QiskitDevice._execute_sampler() calls
`sampler.run(...).result()[0]` as a single chained expression
The only way to keep the Job is to override _execute_sampler
and reproduce its body up to the point where it currently discards the job.

This file exists specifically to isolate that coupling in one place, away
from the QAOA solve loop in Pennylane_solver.py: IBMHardwareDevice overrides
ONLY _execute_sampler (copied from pennylane_qiskit==PINNED_PENNYLANE_QISKIT_VERSION,
plus job retention and a pre-submission QPU-time estimate), and delegates
everything else (circuit compilation, parameter binding, result parsing,
the Estimator/expval path used by optimization=True) to the base class
unchanged.

Caveat: if pennylane_qiskit is upgraded and _execute_sampler's
implementation changes, this override can silently drift out of sync.
_check_pennylane_qiskit_version() only warns when a *different*
pennylane_qiskit version is installed — it cannot detect a same-version
behavioral change, since none is expected within a version.
"""

import pennylane_qiskit
from pennylane_qiskit.qiskit_device import QiskitDevice, circuit_to_qiskit
from qiskit_ibm_runtime import SamplerV2 as Sampler

from quantum.utils.logger import get_logger

PINNED_PENNYLANE_QISKIT_VERSION = "0.45.0"

# Heuristic, from exactly two real jobs on ibm_marrakesh/ibm_fez (see
# quantum/hardware/README.md's "IBM quota" section): billed job.usage() in
# both cases matched round(EMPIRICAL_JOB_BASELINE_SEC + gate-model estimate)
# to the second. Not an IBM-documented rule — a fixed per-job floor
# (dispatch, compilation, queue admission) is a plausible explanation, but
# n=2 is not enough to be confident this generalizes. Revisit as more real
# billed_usage data accumulates.
EMPIRICAL_JOB_BASELINE_SEC = 2.0


def _check_pennylane_qiskit_version():
    installed = pennylane_qiskit.__version__
    if installed != PINNED_PENNYLANE_QISKIT_VERSION:
        get_logger().minimal(
            "⚠️  IBMHardwareDevice._execute_sampler was copied from "
            f"pennylane_qiskit=={PINNED_PENNYLANE_QISKIT_VERSION}; installed "
            f"version is {installed}. If pennylane_qiskit's own "
            "_execute_sampler changed since, this override may be out of "
            "sync — diff it against the installed package before trusting "
            "results."
        )


class IBMHardwareDevice(QiskitDevice):
    """
    QiskitDevice subclass for real IBM hardware (device="qiskit.remote")
    that retains the submitted Sampler Job so job_id and job.usage()/
    job.metrics() are available after each shot-sampling call, and logs a
    pre-execution QPU-time estimate (see quantum.hardware.qpu_time_estimate
    and quantum.hardware.qpu_clops) before submitting.

    Construct directly — not via qml.device("qiskit.remote", ...), which
    would give you the stock QiskitDevice instead:

        dev = IBMHardwareDevice(wires=circuit_wires, backend=backend, session=session)

    Attributes set by each _execute_sampler call (None until the first one
    runs):
        last_job: the qiskit_ibm_runtime Job for the most recent sample.
        last_job_id: str job ID of the most recent sample.
        last_gate_estimate: dict from qpu_time_estimate.estimate_qpu_time(),
            or None if that model failed.
        last_clops_estimate: dict from
            qpu_time_estimate.estimate_qpu_time_clops(), or None if no
            published CLOPS was available or that model failed.
    """

    def __init__(self, *args, **kwargs):
        _check_pennylane_qiskit_version()
        super().__init__(*args, **kwargs)
        self.logger = get_logger()
        self.last_job = None
        self.last_job_id = None
        self.last_shots = None
        self.last_gate_estimate = None
        self.last_clops_estimate = None

    def _execute_sampler(self, circuit, session):
        """
        Copy of pennylane_qiskit.qiskit_device.QiskitDevice._execute_sampler
        (see PINNED_PENNYLANE_QISKIT_VERSION in the module docstring),
        modified only to log a pre-submission QPU-time estimate and retain
        the Job before calling .result() on it.
        """
        qcirc = [
            circuit_to_qiskit(circuit, self.num_wires, diagonalize=True, measure=True)
        ]
        sampler = Sampler(mode=session) if session else Sampler(mode=self.backend)
        compiled_circuits = self.compile_circuits(qcirc)
        sampler.options.update(**self._kwargs)

        shots = circuit.shots.total_shots if circuit.shots.total_shots else None
        self.last_shots = shots
        self._log_qpu_time_estimate(compiled_circuits[0], shots)

        job = sampler.run(compiled_circuits, shots=shots)
        self.last_job = job
        self.last_job_id = job.job_id()
        self.logger.minimal(f"🆔 IBM Job ID: {self.last_job_id}")

        # len(compiled_circuits) is always 1 so the indexing does not matter.
        result = job.result()[0]
        classical_register_name = compiled_circuits[0].cregs[0].name
        self._current_job = getattr(result.data, classical_register_name)

        self._samples = self.generate_samples(0)
        res = [
            mp.process_samples(self._samples, wire_order=self.wires)
            for mp in circuit.measurements
        ]

        single_measurement = len(circuit.measurements) == 1
        return (res[0],) if single_measurement else tuple(res)

    def _log_qpu_time_estimate(self, compiled_circuit, shots):
        """
        Pre-execution estimate via quantum.hardware.qpu_time_estimate (gate
        model + CLOPS model — see that module's docstring for why neither is
        exact and both are logged rather than a single number). Never
        raises — a failed model is logged and skipped, since this is a
        diagnostic add-on, not part of the execution path.
        """
        from quantum.hardware.qpu_clops import get_backend_clops
        from quantum.hardware.qpu_time_estimate import (
            estimate_qpu_time,
            estimate_qpu_time_clops,
        )

        self.last_gate_estimate = None
        self.last_clops_estimate = None

        try:
            gate_estimate = estimate_qpu_time(compiled_circuit, self.backend, shots)
            self.last_gate_estimate = gate_estimate
            self.logger.standard(
                f"⏱️  Gate-model estimate: {gate_estimate['total_estimate_sec']:.3f}s "
                f"({shots} shots, depth {gate_estimate['depth']}, "
                f"{sum(gate_estimate['gate_counts'].values())} native gates)"
            )
        except Exception as exc:
            self.logger.standard(f"(non-fatal) Gate-model QPU estimate failed: {exc}")

        try:
            clops = get_backend_clops(self.backend.name)
            if clops and clops.get("value"):
                clops_estimate = estimate_qpu_time_clops(
                    compiled_circuit, self.backend, shots, clops["value"]
                )
                self.last_clops_estimate = clops_estimate
                self.logger.standard(
                    f"⏱️  CLOPS-model estimate: {clops_estimate['total_estimate_sec']:.3f}s "
                    f"(CLOPS={clops['value']}, 2Q-depth={clops_estimate['two_qubit_depth']})"
                )
            else:
                self.logger.standard(
                    f"(info) No published CLOPS for backend '{self.backend.name}'"
                )
        except Exception as exc:
            self.logger.standard(f"(non-fatal) CLOPS-model QPU estimate failed: {exc}")

        if self.last_gate_estimate is not None:
            gate_sec = self.last_gate_estimate["total_estimate_sec"]
            predicted = round(EMPIRICAL_JOB_BASELINE_SEC + gate_sec)
            self.logger.minimal(
                f"💵 Predicted billed cost: ~{predicted}s "
                f"(heuristic: round({EMPIRICAL_JOB_BASELINE_SEC:.1f}s baseline + "
                f"{gate_sec:.2f}s gate estimate); n=2 calibration points, not an "
                "IBM-documented rule — see quantum/hardware/README.md)"
            )

    def get_usage(self):
        """
        job.usage() and job.metrics() for the most recent sampler job, for
        comparison against last_gate_estimate/last_clops_estimate. Also
        records this (estimate, billed) pair into the persistent IBM usage
        log — see quantum.hardware.qpu_calibration.record_ibm_usage — so
        EMPIRICAL_JOB_BASELINE_SEC can be checked against real accumulated
        data over time, across sessions, not just this one run.

        None if no job has run yet, or if job.usage() itself fails (e.g. the
        plan doesn't support usage reporting) — a failure recording the
        calibration log entry is non-fatal and doesn't affect the return
        value, since that's a diagnostic add-on, not part of the result.
        """
        if self.last_job is None:
            return None
        try:
            usage_info = {
                "usage": self.last_job.usage(),
                "metrics": self.last_job.metrics(),
            }
        except Exception as exc:
            self.logger.standard(f"(non-fatal) Could not retrieve job usage: {exc}")
            return None

        try:
            from quantum.hardware.qpu_calibration import record_ibm_usage

            record_ibm_usage(
                self.backend.name,
                shots=self.last_shots,
                gate_estimate_sec=(
                    self.last_gate_estimate["total_estimate_sec"]
                    if self.last_gate_estimate
                    else None
                ),
                clops_estimate_sec=(
                    self.last_clops_estimate["total_estimate_sec"]
                    if self.last_clops_estimate
                    else None
                ),
                billed_usage_sec=usage_info["usage"],
            )
        except Exception as exc:
            self.logger.standard(f"(non-fatal) Could not record IBM usage calibration: {exc}")

        return usage_info
