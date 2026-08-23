"""
Pre-execution estimates of real IBM QPU compute time. Two independent models:

  estimate_qpu_time() — a calibration-based physics model, from the
    transpiled circuit's native gate count/depth and the backend's live
    calibration data (gate durations, readout duration, active-reset
    latency, repetition delay). See quantum/benchmark/qpu_test.py for the
    raw calibration fields this builds on.

  estimate_qpu_time_clops() — IBM's published CLOPS (Circuit Layer
    Operations Per Second) throughput benchmark, applied to this circuit's
    two-qubit-gate depth. See quantum.hardware.qpu_clops.get_backend_clops().

Neither is exact, and they fail in different directions. The physics model
only sees the QPU's own dt-clock timeline (gates, reset, readout, rep_delay)
— it has no way to see the classical control loop wrapped around each shot
(parameter binding, primitive dispatch/feedback, results retrieval), so it
tends to undershoot. CLOPS is measured empirically end-to-end so it *does*
fold that overhead in, but it's anchored to IBM's own fixed benchmark circuit
shape (random SU(4) layers at the backend's quantum-volume depth), not this
circuit's actual gate mix — and like the physics model, it has no way to see
one-time per-job overhead (server-side compilation, queue admission) that
doesn't scale with shots. Both are pre-execution predictions, not billing —
compare them against actual job.usage()/job.metrics() once a job completes.

Both are active-QPU-execution-time estimates only. Neither predicts queue
wait, which depends on backend load at submission time.
"""

from qiskit import transpile


def _max_duration(target, op_name, physical_qubits):
    """Max calibrated duration for op_name across physical_qubits, skipping
    any qubit whose calibration is missing (can happen after a partial
    recalibration) rather than failing the whole estimate over one qubit."""
    durations = [
        target[op_name][(q,)].duration
        for q in physical_qubits
        if target[op_name][(q,)].duration is not None
    ]
    if not durations:
        raise ValueError(
            f"No timing calibration for '{op_name}' on any of the qubits "
            f"used by this circuit ({physical_qubits})."
        )
    return max(durations)


def estimate_qpu_time(circuit, backend, shots, active_reset=True):
    """
    Estimate real QPU execution time for `circuit` on `backend`, in seconds.

    Args:
        circuit: A Qiskit QuantumCircuit to be run on `backend` (need not
            already be transpiled — this transpiles it itself).
        backend: A Qiskit BackendV2 with `.target` and `.dt` (a real hardware
            backend from QiskitRuntimeService; simulators without timing
            calibration aren't supported).
        shots: Number of shots the job will run.
        active_reset: Whether the run uses active qubit reset between shots
            (IBM Runtime's default, `execution.init_qubits=True`) rather than
            relying solely on `rep_delay` for passive decay.

    Returns:
        dict: {
            "total_estimate_sec": float,
            "per_shot_sec": float,
            "circuit_duration_sec": float,
            "init_latency_sec": float,
            "readout_latency_sec": float,
            "rep_delay_sec": float,
            "shots": int,
            "gate_counts": {gate_name: count, ...},  # native gates, post-transpile
            "depth": int,                             # native-gate depth, post-transpile
            "num_qubits_used": int,
            "backend_name": str,
        }

    Raises:
        ValueError: If the backend has no timing calibration data for the
            scheduled circuit (e.g. it's a non-timed simulator).
    """
    scheduled = transpile(
        circuit, backend=backend, optimization_level=1, scheduling_method="alap"
    )

    if scheduled.duration is None:
        raise ValueError(
            f"Backend '{backend.name}' returned no timing information for the "
            "scheduled circuit — cannot estimate QPU time (real hardware "
            "backends from QiskitRuntimeService should always provide this)."
        )

    dt = backend.dt
    circuit_duration_sec = scheduled.duration * dt

    if scheduled.layout is not None:
        physical_qubits = scheduled.layout.final_index_layout()
    else:
        physical_qubits = list(range(scheduled.num_qubits))

    init_latency_sec = (
        _max_duration(backend.target, "reset", physical_qubits) if active_reset else 0.0
    )
    readout_latency_sec = _max_duration(backend.target, "measure", physical_qubits)
    rep_delay_sec = getattr(backend, "default_rep_delay", None) or 0.0

    per_shot_sec = (
        init_latency_sec + circuit_duration_sec + readout_latency_sec + rep_delay_sec
    )
    total_estimate_sec = per_shot_sec * shots

    return {
        "total_estimate_sec": total_estimate_sec,
        "per_shot_sec": per_shot_sec,
        "circuit_duration_sec": circuit_duration_sec,
        "init_latency_sec": init_latency_sec,
        "readout_latency_sec": readout_latency_sec,
        "rep_delay_sec": rep_delay_sec,
        "shots": shots,
        "gate_counts": dict(scheduled.count_ops()),
        "depth": scheduled.depth(),
        "num_qubits_used": len(physical_qubits),
        "backend_name": backend.name,
    }


def estimate_qpu_time_clops(circuit, backend, shots, clops):
    """
    Estimate real QPU execution time from IBM's published CLOPS for
    `backend`, rather than from calibration data.

    IBM measures CLOPS by running a fixed benchmark (M=100 circuit
    templates x K=10 parameter updates x S=100 shots, over circuits with
    two-qubit-gate layer depth D equal to the backend's quantum-volume
    depth): CLOPS = M*K*S*D / time_taken. Applying it to an arbitrary
    circuit inverts that relationship for a single template/parameter set:

        total_estimate_sec = shots * two_qubit_depth / clops

    where two_qubit_depth is *this* circuit's own two-qubit-gate layer depth
    (post-transpile) — the same quantity D represents in IBM's benchmark.

    Args:
        circuit: A Qiskit QuantumCircuit (need not be pre-transpiled).
        backend: A Qiskit BackendV2 — used only to transpile for the correct
            coupling map/basis gates, not for calibration data.
        shots: Number of shots the job will run.
        clops: CLOPS value for this backend, e.g. from
            quantum.hardware.qpu_clops.get_backend_clops()["value"].

    Returns:
        dict: {
            "total_estimate_sec": float,
            "two_qubit_depth": int,
            "shots": int,
            "clops": int,
            "backend_name": str,
        }
    """
    transpiled = transpile(circuit, backend=backend, optimization_level=1)
    two_qubit_depth = transpiled.depth(
        filter_function=lambda instr: len(instr.qubits) >= 2
    )
    total_estimate_sec = (shots * two_qubit_depth) / clops

    return {
        "total_estimate_sec": total_estimate_sec,
        "two_qubit_depth": two_qubit_depth,
        "shots": shots,
        "clops": clops,
        "backend_name": backend.name,
    }
