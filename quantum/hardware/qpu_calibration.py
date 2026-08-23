"""
Empirically-measured QPU calibration, persisted across runs — two unrelated
schemas living in one module because they solve the same underlying problem
(a static model going stale) for the two vendors in different ways:

IQM (record_execution / get_empirical_rate): where a backend publishes no
throughput benchmark (IQM only documents CLOPS for Garnet; most other
machines have none), successive solve() calls build up their own
shots*depth/time rate from *real* completed jobs' measured execution time —
a direct, usable substitute for a missing published CLOPS value in
estimate_qpu_time_clops(). execution_time_sec here comes from the job's own
timeline (execution_started -> execution_ended — see
iqm.iqm_server_client.iqm_server_client.IQMServerClientJob.find_timeline_entry),
ground truth, not an estimate.

IBM (record_ibm_usage / get_average_ibm_overhead): IBM's billed job.usage()
is ground truth too, but unlike IQM there's no missing-CLOPS gap to fill —
IBM already publishes CLOPS and we already have a gate-duration model. What
this accumulates instead is (prediction, actual) pairs, so
ibm_device.py's EMPIRICAL_JOB_BASELINE_SEC heuristic — currently a constant
derived from just two real jobs — can be checked and eventually refit
against real accumulated data rather than staying frozen at that.
"""

import json
from pathlib import Path

DEFAULT_PATH = Path("results") / "qpu_calibration.json"
IBM_USAGE_LOG_PATH = Path("results") / "ibm_usage_log.json"


def record_execution(
    backend_name,
    shots,
    two_qubit_depth,
    execution_time_sec,
    path=DEFAULT_PATH,
    extra_timing=None,
):
    """
    Append one real (shots, depth, execution_time_sec) data point for
    backend_name to the calibration store at `path`.

    Args:
        extra_timing: optional dict of additional measured timing segments
            to store alongside the core data point — e.g. IQM's
            {"compile_time_sec": ..., "validation_time_sec": ...,
            "job_total_sec": ...} (see IQMHardwareBackend). Not used in the
            rate calculation below: rate is meant to measure QPU throughput,
            and compile/validation are classical overhead, not that.

    Returns:
        float | None: this data point's own rate (shots*depth/time), or None
        if execution_time_sec/two_qubit_depth is zero or missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    store = _load(path)
    rate = (
        (shots * two_qubit_depth) / execution_time_sec
        if execution_time_sec and two_qubit_depth
        else None
    )
    entry = {
        "shots": shots,
        "two_qubit_depth": two_qubit_depth,
        "execution_time_sec": execution_time_sec,
        "rate_layers_per_sec": rate,
    }
    if extra_timing:
        entry.update(extra_timing)
    store.setdefault(backend_name, []).append(entry)
    path.write_text(json.dumps(store, indent=2))
    return rate


def get_empirical_rate(backend_name, path=DEFAULT_PATH):
    """
    Average shots*depth/execution_time_sec rate observed for backend_name
    across all recorded runs, or None if none are recorded yet.
    """
    store = _load(Path(path))
    rates = [
        e["rate_layers_per_sec"]
        for e in store.get(backend_name, [])
        if e.get("rate_layers_per_sec")
    ]
    if not rates:
        return None
    return sum(rates) / len(rates)


def record_ibm_usage(
    backend_name,
    shots,
    gate_estimate_sec,
    clops_estimate_sec,
    billed_usage_sec,
    path=IBM_USAGE_LOG_PATH,
):
    """
    Append one real (gate estimate, CLOPS estimate, billed usage) data point
    for backend_name to the IBM usage log at `path`.

    Unlike record_execution() above, nothing here is measured by us —
    gate_estimate_sec/clops_estimate_sec are our own pre-execution
    predictions and billed_usage_sec is IBM's own job.usage() number, not
    something independently verified. The point of logging is purely to
    accumulate (prediction, actual) pairs over many separate runs/sessions —
    a per-session benchmark summary (BenchmarkRunner's
    total_billed_qpu_usage_sec) gets overwritten by the next session's own
    JSON file; this doesn't.

    Args:
        gate_estimate_sec: IBMHardwareDevice.last_gate_estimate's
            total_estimate_sec, or None if that model failed for this window.
        clops_estimate_sec: same, from last_clops_estimate, or None.
        billed_usage_sec: job.usage() for this window.

    Returns:
        float | None: billed_usage_sec - gate_estimate_sec for this data
        point (the empirical overhead the round(2s + gate) heuristic is
        trying to approximate), or None if gate_estimate_sec is unavailable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    store = _load(path)
    overhead_sec = (
        billed_usage_sec - gate_estimate_sec
        if gate_estimate_sec is not None and billed_usage_sec is not None
        else None
    )
    store.setdefault(backend_name, []).append(
        {
            "shots": shots,
            "gate_estimate_sec": gate_estimate_sec,
            "clops_estimate_sec": clops_estimate_sec,
            "billed_usage_sec": billed_usage_sec,
            "overhead_sec": overhead_sec,
        }
    )
    path.write_text(json.dumps(store, indent=2))
    return overhead_sec


def get_average_ibm_overhead(backend_name, path=IBM_USAGE_LOG_PATH):
    """
    Average (billed_usage_sec - gate_estimate_sec) observed for backend_name
    across all recorded runs, or None if none are recorded yet. This is the
    empirical version of ibm_device.py's hardcoded
    EMPIRICAL_JOB_BASELINE_SEC — not wired in automatically (that constant
    is still what's actually used for the predicted-cost log line), just
    available to check it against as real data accumulates.
    """
    store = _load(Path(path))
    overheads = [
        e["overhead_sec"]
        for e in store.get(backend_name, [])
        if e.get("overhead_sec") is not None
    ]
    if not overheads:
        return None
    return sum(overheads) / len(overheads)


def iqm_two_qubit_depth(instructions):
    """
    Two-qubit-gate layer depth of a raw IQM circuit's instruction sequence
    (job._iqm_job.payload()[0][i].instructions — the actual native gates
    that ran on hardware, post server-side compilation).

    Rewritten (not imported) from IQM's own iqm-benchmarks methodology —
    iqm.benchmarks.utils.count_2q_layers there does the equivalent
    computation via circuit_to_dag()+dag.layers() over a Qiskit circuit;
    that package is a full benchmark suite (pulls in matplotlib, xarray) not
    worth depending on for one function, and its input type (Qiskit
    QuantumCircuit) doesn't match what we have here anyway (IQM's own flat,
    already-linearized native instruction sequence, with no Qiskit DAG
    involved). This walks that sequence directly: only instructions whose
    locus spans >=2 qubits advance the per-qubit depth counters — the same
    longest-path-per-layer idea, applied to the format we actually have.
    """
    qubit_depth = {}
    max_depth = 0
    for instr in instructions:
        locus = instr.locus if hasattr(instr, "locus") else instr["locus"]
        if len(locus) < 2:
            continue
        layer = max(qubit_depth.get(q, 0) for q in locus) + 1
        for q in locus:
            qubit_depth[q] = layer
        max_depth = max(max_depth, layer)
    return max_depth


def _load(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())
