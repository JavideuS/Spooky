"""
Thin wrapper around Qrisp's IQMBackend adding job telemetry: job_id logging
and empirical QPU-time calibration recording (see
quantum.hardware.qpu_calibration), extracted out of Pennylane_solver.py so
hardware-accounting concerns live in one dedicated file instead of mixed
into the QAOA solve loop.

Bypasses the higher-level IQMBackend.run() (which internally does
run_async(...).result().all_counts and discards the Job — see
IQMBackend.run() in iqm/qrisp_iqm/backends/backend.py) so the Job handle
survives long enough to read job_id, and afterwards to read its own
timeline — execution/compile/validation/job-total segments, ground truth
rather than a model — and native gate sequence (via job._iqm_job.payload())
for calibration.
"""

from quantum.hardware.qpu_calibration import iqm_two_qubit_depth, record_execution
from quantum.utils.logger import get_logger


class IQMHardwareBackend:
    """
    Args:
        backend: A Qrisp iqm.qrisp_iqm.IQMBackend instance.
    """

    def __init__(self, backend):
        self.backend = backend
        self.logger = get_logger()
        self.last_job_id = None
        # Set by _record_execution: {"execution_time_sec", "compile_time_sec",
        # "validation_time_sec", "job_total_sec"} (keys present only when that
        # timeline entry existed), or None if recording failed/was skipped.
        self.last_timing = None

    def run(self, circuit, shots):
        """
        Submit circuit (a Qrisp circuit) for shots shots, block for results,
        and return a Qiskit/Qrisp-style bitstring -> count dict. Logs
        job_id immediately after submission, and after completion records
        the job's real measured timing into last_timing and the calibration
        store (non-fatal on failure — see _record_execution).
        """
        job = self.backend.run_async(circuit, shots=shots)
        self.last_job_id = job.job_id
        self.logger.minimal(f"🆔 IQM Job ID: {self.last_job_id}")
        counts = job.result().get_counts()

        self.last_timing = None
        self._record_execution(job, shots)
        return counts

    def _record_execution(self, job, shots):
        """
        Pulls real timeline segments and the actual native gate sequence off
        the completed job, computes two-qubit-gate depth, and persists them
        via quantum.hardware.qpu_calibration.record_execution(). Reaches
        into job._iqm_job (the wrapped iqm_client.CircuitJob) because
        Qrisp's IQMJob doesn't expose timeline/payload at its own public API
        level — confirmed this is the same approach IQM's own official
        iqm-benchmarks package uses internally (see
        iqm.benchmarks.utils.retrieve_all_job_metadata, which reaches into
        the identical j._iqm_job.data.timeline). Never raises — failures are
        logged and skipped, since this is a diagnostic add-on, not part of
        the solve path.

        Captures four segments — mirrors iqm-benchmarks' own
        retrieve_clops_elapsed_times (job_total/compile_total/execution_total)
        plus validation, which the timeline shows separately from compilation:
            execution_time_sec: execution_started -> execution_ended — the
                only one that's real QPU device time; the only one used in
                qpu_calibration's rate calculation.
            compile_time_sec: compilation_started -> compilation_ended —
                classical, server-side circuit compilation.
            validation_time_sec: validation_started -> validation_ended —
                in practice this segment tends to be near-instant on an idle
                device, so a large value here is more likely queue/wait time
                on a busy one than genuine validation computation.
            job_total_sec: received -> ready — the full server-side span
                covering all of the above, for context.
        """
        try:
            raw_job = job._iqm_job

            execution_time_sec = self._timeline_segment(
                raw_job, "execution_started", "execution_ended"
            )
            if execution_time_sec is None:
                self.logger.standard(
                    "(info) IQM job timeline missing execution_started/"
                    "execution_ended — skipping calibration record."
                )
                return

            compile_time_sec = self._timeline_segment(
                raw_job, "compilation_started", "compilation_ended"
            )
            validation_time_sec = self._timeline_segment(
                raw_job, "validation_started", "validation_ended"
            )
            job_total_sec = self._timeline_segment(raw_job, "received", "ready")

            circuits, _params = raw_job.payload()
            two_qubit_depth = iqm_two_qubit_depth(circuits[0].instructions)

            extra_timing = {
                k: v
                for k, v in {
                    "compile_time_sec": compile_time_sec,
                    "validation_time_sec": validation_time_sec,
                    "job_total_sec": job_total_sec,
                }.items()
                if v is not None
            }
            self.last_timing = {"execution_time_sec": execution_time_sec, **extra_timing}

            rate = record_execution(
                self.backend.name,
                shots,
                two_qubit_depth,
                execution_time_sec,
                extra_timing=extra_timing or None,
            )
            rate_str = f", rate {rate:.0f} layers/s" if rate else ""
            segments = [f"execution={execution_time_sec:.4f}s"]
            if compile_time_sec is not None:
                segments.append(f"compile={compile_time_sec:.4f}s")
            if validation_time_sec is not None:
                segments.append(f"validation={validation_time_sec:.4f}s")
            if job_total_sec is not None:
                segments.append(f"job_total={job_total_sec:.4f}s")
            self.logger.standard(
                f"📏 Real IQM timing — {', '.join(segments)} "
                f"({shots} shots, 2Q-depth {two_qubit_depth}{rate_str})"
            )
        except Exception as exc:
            self.logger.standard(f"(non-fatal) IQM execution calibration failed: {exc}")

    def _timeline_segment(self, raw_job, start_status, end_status):
        """
        Seconds between two named timeline entries on raw_job
        (job._iqm_job), or None if either is missing from this job's
        timeline (a status IQM didn't report for this particular job).
        """
        start = raw_job.find_timeline_entry(start_status)
        end = raw_job.find_timeline_entry(end_status)
        if start is None or end is None:
            return None
        return (end.timestamp - start.timestamp).total_seconds()
