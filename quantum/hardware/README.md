# Hardware Telemetry

Real quantum hardware runs cost money and consume quota. This package exists to make that cost visible *before* you spend it — pre-execution time estimates, real job-ID/usage capture, and (for IQM) empirical calibration built from your own completed jobs.

**If you're about to run a real hardware sweep, read the "IBM quota" section below first.**

## Setup

`qpu_clops.py`'s IBM Cloud REST call (fetching published CLOPS) needs `IBM_TOKEN` (an IAM API key) and `IBM_CRN` (your Service CRN) in `.env` or the environment. Without them, `get_backend_clops()` raises and `IBMHardwareDevice` just logs `(info) No published CLOPS for backend '...'` and skips the CLOPS model — everything else (gate model, job telemetry, the predicted-cost heuristic) still works without these credentials. `qiskit_ibm_runtime`'s own IBM Quantum Platform credentials (used for the actual solve, via `QiskitRuntimeService()`) are separate and unrelated to these.

## Package Structure

- **`ibm_device.py`** — `IBMHardwareDevice`, a `pennylane_qiskit.QiskitDevice` subclass for `device="qiskit.remote"`. Logs a pre-submission time estimate, retains the submitted `Job` (job_id, `.usage()`/`.metrics()`) that the stock device discards, and now also logs a single predicted billed cost — see below.
- **`ibm_session.py`** — `IBMSessionManager`, holds one IBM Runtime `Session` open across a whole `solve()` call's windowed loop so only the first window queues.
- **`iqm_backend.py`** — `IQMHardwareBackend`, wraps Qrisp's `IQMBackend` to capture job_id and the completed job's real timeline (execution/compile/validation/job-total) instead of discarding it.
- **`qpu_time_estimate.py`** — the two pre-execution models: `estimate_qpu_time()` (gate-duration/calibration-based) and `estimate_qpu_time_clops()` (throughput-based, from a published CLOPS number).
- **`qpu_clops.py`** — fetches IBM's published CLOPS per backend via the IBM Cloud REST API (needs `IBM_TOKEN`/`IBM_CRN` in `.env`; not exposed on the `BackendV2` Python object itself).
- **`qpu_calibration.py`** — persists real, measured (not modeled) data across sessions, two unrelated schemas for the two vendors: IQM's `results/qpu_calibration.json` (execution time per backend, builds an empirical throughput rate — IQM has no real substitute otherwise, only Garnet publishes CLOPS) and IBM's `results/ibm_usage_log.json` (every real job's `(gate_estimate, CLOPS_estimate, billed_usage)` triple, logged automatically by `IBMHardwareDevice.get_usage()` — not yet fed back into anything automatically, see "IBM quota" below).

## IBM quota: what we've learned from real runs

Two models estimate QPU time *before* you submit:

- **Gate model** (`estimate_qpu_time`): scheduled circuit duration (from the backend's live gate/readout/reset calibration) + init + readout + rep_delay, × shots.
- **CLOPS model** (`estimate_qpu_time_clops`): `shots × your_circuit's_2Q_depth / published_CLOPS`.

Neither matches what IBM actually bills (`job.usage()`) directly — both only see active-QPU-adjacent time, not the fixed per-job overhead (compilation dispatch, queue admission) that a short job pays disproportionately for. From two real runs on `ibm_marrakesh`/`ibm_fez`:

| shots | gate est. | CLOPS est. | billed usage |
| ----- | --------- | ---------- | ------------ |
| 500   | 0.13s     | 0.12s      | 2s           |
| 10000 | 2.63s     | 1.40s      | 5s           |

**Finding 1 — CLOPS underestimates for QAOA-shaped circuits.** IBM's published CLOPS is CLOPS_V: calibrated against a Quantum-Volume-style benchmark circuit with fixed depth = num_qubits and *every* layer densely packed with 2-qubit gates across roughly half the qubits. A QAOA ansatz on a sparse problem graph, routed through IBM's heavy-hex coupling map, has the same *layer count* but much lower gate density per layer — CLOPS's rate implicitly assumes the QV template's density, so it under-predicts time for circuits like ours. This isn't a bug in our formula; it's a known generalization limit of reusing a benchmark calibrated on one circuit shape for a structurally different one. The gate model doesn't have this problem — it's built from the circuit's own scheduled durations, not someone else's calibration circuit.

**Finding 2 — billed usage looks like `round(2s + gate-model estimate)`.** Subtracting a ~2s constant from both billed values above and comparing to the gate model's own estimate:

| shots | gate est. | 2s + gate est. | rounded | actual billed |
| ----- | --------- | -------------- | ------- | ------------- |
| 500   | 0.13s     | 2.13s          | 2s      | 2s            |
| 10000 | 2.63s     | 4.63s          | 5s      | 5s            |

Both match exactly *after rounding* — but the two runs were on different backends (`ibm_fez` and `ibm_marrakesh`), and `results/ibm_usage_log.json`'s per-backend `overhead_sec` (`billed_usage_sec - gate_estimate_sec`, logged automatically now by every real run — see `record_ibm_usage`/`get_average_ibm_overhead` in `qpu_calibration.py`) shows they aren't actually the same number: 1.869s for `ibm_fez`, 2.37s for `ibm_marrakesh`. Rounding to whole seconds happened to hide that ~0.5s gap in both cases. **This is a hypothesis from two data points on two different backends, not a documented IBM billing rule** — the single shared constant `EMPIRICAL_JOB_BASELINE_SEC = 2.0` is a simplification, not confirmation the floor is backend-independent. Treat it as a working estimate, revisit — with real per-backend numbers, not more manual table-building — as `ibm_usage_log.json` accumulates.

Given this, `IBMHardwareDevice` now logs a single predicted-cost line per submission:

```
💵 Predicted billed cost: ~2s (heuristic: round(2.0s baseline + 0.13s gate estimate); n=2 calibration points, not an IBM-documented rule)
```

Compare it against the `💰 Billed QPU usage:` line that appears after the job completes to sanity-check the heuristic on your own workload.

## IQM timing

`IQMHardwareBackend` records four segments from every completed job's timeline (IQM has no billing API to compare against, so this is the ground truth itself, not an estimate):

- `execution_time_sec` — `execution_started` → `execution_ended`, real QPU device time. The only segment used in `qpu_calibration.py`'s empirical rate (compile/validation are classical overhead, not QPU throughput).
- `compile_time_sec` — `compilation_started` → `compilation_ended`, server-side classical compilation.
- `validation_time_sec` — `validation_started` → `validation_ended`. In practice this is near-instant (~1ms) on an idle device — a large value here on a busy one likely reflects queue wait rather than genuine validation work.
- `job_total_sec` — `received` → `ready`, the full server-side span.

All four are appended to `results/qpu_calibration.json` per backend after every real IQM job, independent of any model.

## Build / execution / queue / overhead split

`BenchmarkRunner` reports four separate times per run instead of one lumped "execution time":

- `build_time_sec` — QUBO construction (`builder.build()`), before the solver ever runs.
- `execution_time_sec` — **the real-execution reference itself**, summed across windows: IBM's billed `usage` where available, else IQM's `job_total_sec` (its own server-processing span), else the gate-model estimate as a last resort.
- `queue_time_sec` — **hardware runs only.** Per window: wall-clock time around the hardware call (`wall_clock_sec`, `time.time()` before/after the sampling call) minus that same real-execution reference. Summed across all windows in the run.
- `overhead_sec` — everything else in `solve_duration` that isn't either of the above: backend/session setup (e.g. searching for a `least_busy` backend, negotiating an IBM Runtime Session), and classical per-window preprocessing (BFS/diagonal-fixing) and post-processing (decode, path merging). `solve_duration ≈ execution_time_sec + queue_time_sec + overhead_sec`.

**This replaced an earlier, wrong version** that computed `execution_time_sec = solve_duration - queue_time_sec` — on a real run that logged `💰 Billed QPU usage: 2` it reported `exec=11.00s`, more than 5× the actual billed time, because `solve_duration` includes backend selection, session negotiation, and classical preprocessing that isn't queue time either, and all of it was silently landing in "execution." The fix: `execution_time_sec` is assigned directly from the real reference (the same value queue time is computed against), never derived by subtraction — `overhead_sec` is the honest name for what's left over, instead of hiding it inside "execution."

A run with no computable split (classical solvers, or a windowed hardware run where every window happened to be fully pre-processed and never touched hardware) gets none of these four keys beyond `build_time_sec` — `execution_time_sec` stays as the plain solve duration, as before the split existed.

The per-run log line shows the breakdown once it's available:

```
Run 1: ✅ Valid | Time: build=0.00s, exec=2.00s, queue=24.41s, overhead=9.00s | Est. QPU time (gate=0.13s, clops=0.12s, billed=2.00s) | Energy: -7.0667
```

`BenchmarkRunner`'s summary also totals billed usage, queue time, and overhead across all runs in the benchmark (`total_billed_qpu_usage_sec`, `total_queue_time_sec`, `total_overhead_sec`), printed as `Total billed QPU usage this benchmark: ...`.

## IBM empirical calibration

`IBMHardwareDevice.get_usage()` now logs every real job's `(shots, gate_estimate_sec, clops_estimate_sec, billed_usage_sec, overhead_sec)` to `results/ibm_usage_log.json` per backend, via `record_ibm_usage()` — this is separate from (and doesn't replace) `BenchmarkRunner`'s per-session `total_billed_qpu_usage_sec`, which only summarizes one session's own JSON file and gets superseded by the next session's. `get_average_ibm_overhead(backend_name)` reads it back.

**Not yet wired into anything automatically** — `EMPIRICAL_JOB_BASELINE_SEC = 2.0` in `ibm_device.py` is still the hardcoded constant actually used for the predicted-cost log line. This log exists so that constant can eventually be checked and refit per-backend against real accumulated data (see Finding 2 above — overhead already looks backend-dependent, not the single shared number the constant currently assumes) rather than staying frozen at the two points it started from.

## Ideas for later (not yet built)

- **IQM CLOPS per machine**: IQM only publicly documents CLOPS for Garnet. `iqm-benchmarks` (`iqm.benchmarks.quantum_volume.clops.CLOPSBenchmark`) can compute an official CLOPS_V/CLOPS_H for any machine by actually running the benchmark — `quantum/hardware/iqm_clops_calibration.ipynb` is set up to do this. **Attempted 2026-08-22, deferred**: one full-device Garnet run costs ~26 of a 30-credits/month allotment — essentially the entire monthly budget for one run — and Emerald was unavailable at the time regardless. Revisit once credits are specifically budgeted for it; results storage (`results/iqm_clops.json`, matching `qpu_clops.py`'s pattern for IBM) is already built, just unpopulated.
- Auto-refit `EMPIRICAL_JOB_BASELINE_SEC` (or replace it with a per-backend lookup) from `ibm_usage_log.json` once there's enough data to make that meaningful.
