import os
import pennylane as qml
from pennylane import numpy as np
from .base_solver import BaseSolver
from quantum.utils import preprocess as preprocess_modes
from quantum.hardware.ibm_session import IBMSessionManager
import time


class PennylaneSolver(BaseSolver):
    # (name, num_qubits), ascending — smallest machine that fits wins, to
    # spend quota on the cheapest device that can run the window.
    # Sirius is deliberately excluded: it uses resonator-mediated coupling
    # (component 'COMPR1'), which Qrisp's IQM connector can't yet transpile/
    # route for (raises ValueError). Pass machine="sirius" explicitly to try
    # it anyway once Qrisp adds support.
    IQM_MACHINE_TIERS = [
        ("garnet", 20),
        ("emerald", 54),
    ]

    def __init__(
        self,
        normalize_scale=0,
        num_reads="auto",
        layers=2,
        optimizer="GradientDescent",
        opt_steps=10,
        device="default.qubit",
        params=None,
        verbose_level=2,
        machine=None,
        threads=None,
        use_session=True,
        session_max_time=None,
        **kwargs,
    ):
        """
        Args:
            threads: CPU thread count for lightning.qubit's OpenMP backend
                (default.qubit, lightning.gpu unaffected). None (default)
                leaves OpenMP's own default in place, which is every
                available core (os.cpu_count()) — set this to pin a specific
                count, e.g. to match another solver's thread count for a
                fair wall-clock comparison. Must be set before the device is
                constructed: OpenMP reads OMP_NUM_THREADS on first use, not
                on every call, so this only takes effect if set here rather
                than after qml.device(...) has already run once.
            use_session: device="qiskit.remote" only. Holds one IBM Runtime
                Session open across this solve()'s whole windowed loop
                instead of every window's job re-entering the public queue
                from scratch (see quantum.hardware.ibm_session.IBMSessionManager).
                Ignored for any other device. Requires a plan that supports
                Sessions — IBM's Open plan explicitly doesn't; Pay-As-You-Go
                and above do. If opening one fails for any reason, falls
                back to today's per-job-queued behavior with a single
                logged warning rather than raising (never worse than
                use_session=False).
            session_max_time: Forwarded to qiskit_ibm_runtime.Session's
                max_time (seconds, or a string like "2h 30m"). None uses
                IBM's own default (900s).
        """
        super().__init__(
            solver="pennylane",
            normalize_scale=normalize_scale,
            num_reads=num_reads,
            verbose_level=verbose_level,
            layers=layers,
            optimizer=optimizer,
            opt_steps=opt_steps,
            device=device,
            machine=machine,
            threads=threads,
            params=params,
            use_session=use_session,
            session_max_time=session_max_time,
            **kwargs,
        )
        if threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(threads)
        self.optimizer_name = optimizer
        self.p = layers  # Number of QAOA layers
        self.dev = device
        if device == "qiskit.remote":
            from qiskit_ibm_runtime import QiskitRuntimeService

            self.service = QiskitRuntimeService()
        # Fixed backend/machine name (IBM backend or IQM machine), or None to
        # auto-pick (IBM: least_busy; IQM: smallest tier that fits)
        self.machine = machine
        # Backend for qiskit.remote — avoids a least_busy() network call every window
        self.backend = None
        self.hardware_qubits = 0  # actual qubit count of the connected backend
        self._session_manager = IBMSessionManager(
            use_session=use_session, session_max_time=session_max_time
        )
        # Parameters for the QAOA circuit
        self.params = params if params is not None else np.random.rand(layers, 2)
        self.optimizer_steps = opt_steps  # Number of optimization steps
        self.shots = num_reads  # Number of shots for sampling

    def get_shots(self, num_qubits):
        if self.shots == "auto":
            if num_qubits <= 9:
                return 500
            elif num_qubits <= 12:
                return 2500
            elif num_qubits <= 16:
                return 10000  # 15000 safer
            elif num_qubits <= 18:
                return 17000
            else:
                return 25000
        return self.shots

    @classmethod
    def from_config(cls, config):
        """
        Create a PennylaneSolver instance from a configuration dictionary.
        """
        norm_scale = config.get("normalization_scale", 0)
        num_reads = config.get("num_reads", 10)
        layers = config.get("layers", 2)
        optimizer = config.get("optimizer", "GradientDescent")
        device = config.get("device", "default.qubit")
        params = config.get("params", None)
        opt_steps = config.get("optimizer_steps", 10)
        machine = config.get("machine", None)
        threads = config.get("threads", None)
        use_session = config.get("use_session", True)
        session_max_time = config.get("session_max_time", None)
        if optimizer not in ["GradientDescent", "Adam", "QNG", "SPSA", "QNSPSA"]:
            raise ValueError(
                "Optimizer must be either 'GradientDescent', "
                "'QNG', 'SPSA', 'QNSPSA' or 'Adam'"
            )
        if device not in [
            "default.qubit",
            "lightning.qubit",
            "lightning.gpu",
            "qiskit.remote",
            "qiskit.iqm",
        ]:
            raise ValueError(
                "Device must be either 'default.qubit' "
                "or 'lightning' or 'qiskit.remote' or 'qiskit.iqm'"
            )
        return cls(
            normalize_scale=norm_scale,
            num_reads=num_reads,
            layers=layers,
            optimizer=optimizer,
            device=device,
            params=params,
            opt_steps=opt_steps,
            machine=machine,
            threads=threads,
            use_session=use_session,
            session_max_time=session_max_time,
        )

    def _select_iqm_machine(self, num_qubits):
        """
        Pick the cheapest tiered IQM machine that fits num_qubits, ascending by
        qubit count so we don't spend Emerald-tier quota on a Sirius-sized window.
        """
        for name, qubits in self.IQM_MACHINE_TIERS:
            if qubits >= num_qubits:
                return name
        largest_name, largest_qubits = self.IQM_MACHINE_TIERS[-1]
        raise ValueError(
            f"No known IQM machine has >= {num_qubits} qubits "
            f"(largest tiered is '{largest_name}' with {largest_qubits})"
        )

    def _get_backend(self, num_qubits):
        """
        Return a backend for qiskit.remote, refreshing only when the current one
        no longer satisfies the qubit requirement or has not been fetched yet.

        Args:
            num_qubits: Minimum number of qubits required.

        Returns:
            A least-busy IBM backend, or a tiered IQM backend, with at least
            num_qubits qubits.
        """

        if self.backend is None or num_qubits > self.hardware_qubits:
            if self.dev == "qiskit.iqm":
                from dotenv import load_dotenv
                from qrisp.interface import IQMBackend

                # IQM_TOKEN is read from the environment automatically by IQMBackend.
                # Qrisp's native backend, not iqm.qiskit_iqm — PennyLane's sampling
                # path runs the circuit through Qrisp (see _run_iqm_sampler) to avoid
                # an upstream Qiskit bug (memory=True unsupported by IQMBackend,
                # github.com/Qiskit/qiskit/issues/15694) that corrupts BackendSamplerV2
                # results when going through qiskit.remote for IQM.
                load_dotenv()

                machine = self.machine or self._select_iqm_machine(num_qubits)
                self.logger.standard("=" * 60)
                self.logger.standard(f"🔍 Connecting to IQM machine: {machine}")
                backend_start = time.time()
                self.backend = IQMBackend(
                    server_url="https://resonance.iqm.tech",
                    device_instance=machine,
                    use_timeslot=False,
                )
                self.hardware_qubits = self.backend.num_qubits
                backend_time = time.time() - backend_start
                self.logger.standard(
                    f"✓ Backend selected: {machine} ({self.hardware_qubits} qubits)"
                )
                self.logger.standard(f"  Backend selection time: {backend_time:.2f}s")
                self.logger.standard("=" * 60)
            else:  # IBM backend/qiskit.remote
                self.logger.standard("=" * 60)
                backend_start = time.time()
                if self.machine:
                    self.logger.standard(
                        f"🔍 Connecting to IBM backend: {self.machine}"
                    )
                    self.backend = self.service.backend(self.machine)
                    if self.backend.num_qubits < num_qubits:
                        raise ValueError(
                            f"IBM backend '{self.machine}' has only "
                            f"{self.backend.num_qubits} qubits, but this window "
                            f"needs {num_qubits}."
                        )
                else:
                    self.logger.standard(
                        "🔍 Searching for available quantum hardware..."
                    )
                    self.backend = self.service.least_busy(
                        operational=True, simulator=False, min_num_qubits=num_qubits
                    )
                self.hardware_qubits = self.backend.num_qubits
                backend_time = time.time() - backend_start
                self.logger.standard(f"✓ Backend selected: {self.backend.name}")
                self.logger.standard(f"  Backend selection time: {backend_time:.2f}s")
                self.logger.standard("=" * 60)
        else:
            self.logger.standard(f"♻️  Reusing backend: {self.backend.name}")
        return self.backend

    def _pennylane_tape_to_qiskit(self, ansatz_circuit, params, num_qubits):
        """
        Build the ansatz as a PennyLane tape and convert it into a Qiskit
        circuit, via the same conversion qiskit.remote uses internally
        (pennylane_qiskit.circuit_to_qiskit). Used by _pennylane_tape_to_qrisp
        for the IQM bridge. IBMHardwareDevice (quantum.hardware.ibm_device)
        does its own equivalent conversion inline rather than calling this —
        it operates on the tape the qnode hands it directly, not on a
        freshly-built one from ansatz_circuit/params like this method does.
        """
        from pennylane_qiskit.qiskit_device import (
            QISKIT_OPERATION_MAP,
            circuit_to_qiskit,
        )

        with qml.queuing.AnnotatedQueue() as q:
            ansatz_circuit(params)
            qml.sample()
        tape = qml.tape.QuantumScript.from_queue(q)

        # Reduce to the gate set circuit_to_qiskit knows how to map — QAOA layers
        # can produce MultiRZ-style terms that aren't natively in that map.
        (tape,), _ = qml.transforms.decompose(
            tape, stopping_condition=lambda op: op.name in QISKIT_OPERATION_MAP
        )

        return circuit_to_qiskit(tape, num_qubits, diagonalize=True, measure=True)

    def _pennylane_tape_to_qrisp(self, ansatz_circuit, params, num_qubits):
        """
        Build the ansatz as a PennyLane tape and convert it into a Qrisp circuit,
        via the shared Qiskit conversion (_pennylane_tape_to_qiskit), then
        Qiskit -> Qrisp (qrisp.interface.converter.convert_from_qiskit).
        """
        from qrisp.interface.converter import convert_from_qiskit

        qiskit_qc = self._pennylane_tape_to_qiskit(ansatz_circuit, params, num_qubits)
        return convert_from_qiskit(qiskit_qc)

    def _run_iqm_sampler(self, ansatz_circuit, params, num_qubits, shots):
        """
        Run the QAOA ansatz on real IQM hardware via the Qrisp bridge, bypassing
        PennyLane's qiskit.remote Sampler entirely (see _get_backend for why).

        Job telemetry (job_id logging, real-execution-time calibration
        recording) is handled by IQMHardwareBackend — see
        quantum.hardware.iqm_backend — rather than living here, so
        this solve loop doesn't own hardware-accounting concerns directly.

        Returns:
            tuple: (counts, last_timing) — counts is a Qiskit/Qrisp-style
            bitstring -> shot count dict; last_timing is
            IQMHardwareBackend.last_timing after the run (a dict of real
            measured timeline segments, or None if recording failed).
        """
        from quantum.hardware.iqm_backend import IQMHardwareBackend

        qrisp_circuit = self._pennylane_tape_to_qrisp(
            ansatz_circuit, params, num_qubits
        )
        backend = self._get_backend(num_qubits)
        hardware = IQMHardwareBackend(backend)
        counts = hardware.run(qrisp_circuit, shots)
        return counts, hardware.last_timing

    def create_ansatz(self, wires, qaoa_layer):
        """
        Create QAOA ansatz.

        Args:
            wires: Number of qubits
            layers: Number of QAOA layers

        Returns:
            Quantum circuit function
        """

        def ansatz(params):
            # Apply Hadamard to all qubits
            for w in wires:
                qml.Hadamard(wires=w)

            # Apply QAOA layers
            qml.layer(qaoa_layer, self.p, params[0], params[1])

        return ansatz

    def solve(self, builder, optimization=False, preprocess=True):
        """
        Solve QUBO using Pennylane QAOA. Thin wrapper around _solve_impl()
        that guarantees an IBM Runtime Session opened during the solve (see
        quantum.hardware.ibm_session.IBMSessionManager) is always closed
        afterwards — including on an exception or an early `break` out of
        the windowed loop — so a reservation never leaks past this call.
        """
        try:
            return self._solve_impl(
                builder, optimization=optimization, preprocess=preprocess
            )
        finally:
            self._session_manager.close()

    def _solve_impl(self, builder, optimization=False, preprocess=True):
        """
        Args:
            builder: QUBOBuilder instance
            optimization: When True, runs the variational QAOA parameter optimization
                loop before sampling.
            preprocess: When True (default), applies BFS variable reduction,
                diagonal pruning, correction loop, and window stats tracking.
                When False, runs a simple QAOA loop with no preprocessing — useful
                for debugging the raw sampler.

        Returns:
            Dictionary containing solution, energy, and raw response
        """
        best_sample = []
        best_energy = []

        # `preprocess` accepts the mode strings in quantum.utils.preprocess as
        # well as the legacy booleans (True -> "full", False -> "raw").
        mode = preprocess_modes.normalize(preprocess)
        bfs_variant = preprocess_modes.bfs_variant(mode)
        apply_numeric = preprocess_modes.applies_numeric_reduction(mode)

        if not preprocess_modes.uses_windowed_pipeline(mode):
            # Simple loop — no variable reduction, no correction retries
            # Build optimizer once for this run
            if self.optimizer_name == "GradientDescent":
                optimizer = qml.GradientDescentOptimizer()
            elif self.optimizer_name == "QNG":
                optimizer = qml.QNGOptimizer()
            elif self.optimizer_name == "SPSA":
                optimizer = qml.SPSAOptimizer()
            elif self.optimizer_name == "QNSPSA":
                optimizer = qml.QNSPSAOptimizer()
            else:
                optimizer = qml.AdamOptimizer()

            while (builder.total_t) > (builder.current_T):
                if self.norm_scale != 0:
                    builder.Q = self.normalize_qubo(builder.Q, self.norm_scale)

                # get_wires(), NOT range(get_num_wires()). get_wires() returns
                # the QUBO's actual variable indices; get_num_wires() returns
                # only how many there are. Those indices are the global
                # encoding (robot*M*N*T + t*M*N + i*N + j), so they start at 0
                # for the first robot's first window and nowhere else, while
                # qubo_to_ising() builds PauliZ on the raw index. Sizing the
                # device by the count gave a WireError (wires 375-449 absent
                # from a 75-wire device) on anything past that first window.
                # sorted() is defensive only: the pairing with qml.sample()'s
                # column order is self-consistent either way, but set
                # iteration order depends on hash-table layout and the
                # qiskit.remote remap below turns it into a physical qubit
                # assignment, which should not be an accident of that layout.
                wires = sorted(builder.get_wires())
                self.logger.standard(f"Number of qubits: {len(wires)}")

                # An empty window has nothing to sample, and handing a
                # zero-wire device to qml.sample() dies inside custatevec.
                # The preprocess=True path already skips these ("Window fully
                # pre-processed, skipping solver");
                # Doing the same here, but guarding against a window that
                # cannot advance, which would otherwise spin forever.
                if not wires:
                    previous_T = builder.current_T
                    builder.update_problem({})
                    # update_problem() no longer rebuilds Q itself (see its
                    # docstring/comment); this raw preprocess=False loop reads
                    # builder.Q directly, so it must rebuild here.
                    builder.build()
                    if builder.current_T <= previous_T:
                        self.logger.minimal(
                            "Warning: empty window that cannot advance "
                            f"(current_T stuck at {previous_T}) — stopping early"
                        )
                        break
                    continue

                Hc, constant = builder.qubo_to_ising()
                Hmix = qml.qaoa.x_mixer(wires)

                def qaoa_layer(gamma, beta):
                    qml.qaoa.cost_layer(gamma, Hc)
                    qml.qaoa.mixer_layer(beta, Hmix)

                ansatz_circuit = self.create_ansatz(wires, qaoa_layer)
                shots = self.get_shots(len(wires))
                dev = qml.device(self.dev, wires=wires)

                @qml.set_shots(shots)
                @qml.qnode(dev)
                def cost_function(params):
                    ansatz_circuit(params)
                    return qml.expval(Hc)

                if optimization:
                    for step in range(self.optimizer_steps):
                        self.params, cost = optimizer.step_and_cost(
                            cost_function, self.params
                        )
                        if step % 10 == 0:
                            self.logger.debug(f"Step {step}, ⟨H_C⟩ = {cost:.6f}")

                @qml.set_shots(shots)
                @qml.qnode(dev)
                def sample_circuit(params):
                    ansatz_circuit(params)
                    return qml.sample()

                raw_samples = sample_circuit(self.params)
                if raw_samples.ndim == 1:
                    raw_samples = raw_samples.reshape(1, -1)

                samples = []
                energies = []
                for shot_idx in range(min(len(raw_samples), shots)):
                    sample_data = raw_samples[shot_idx]
                    binary_sample = {}
                    for i, wire in enumerate(wires):
                        measurement = (
                            sample_data[i]
                            if hasattr(sample_data, "__getitem__")
                            else sample_data
                        )
                        binary_sample[wire] = int((measurement + 1) // 2)
                    samples.append(binary_sample)
                    energy = sum(
                        builder.Q.get((i, j), 0) * binary_sample[i] * binary_sample[j]
                        for i in wires
                        for j in wires
                    )
                    energies.append(energy)

                if not samples:
                    self.logger.minimal(
                        "Warning: No samples collected, using random sample"
                    )
                    random_sample = {i: np.random.randint(2) for i in wires}
                    samples.append(random_sample)
                    energies.append(
                        sum(
                            builder.Q.get((i, j), 0)
                            * random_sample[i]
                            * random_sample[j]
                            for i in wires
                            for j in wires
                        )
                    )

                best_idx = np.argmin(energies)
                best_sample.append(samples[best_idx])
                best_energy.append(energies[best_idx])

                # update_problem() takes {robot_num: path_segment}, not a
                # position — same construction as the preprocess=True branch.
                # Else it would have only solved first window
                try:
                    path = self.decode_path(
                        samples[best_idx], builder.problem, t_offset=builder.current_T
                    )
                    robot_paths = self.get_robot_paths(path)
                    robot_paths = self._resolve_duplicate_timesteps(
                        robot_paths, builder.problem
                    )
                    builder.update_problem(robot_paths)
                    # update_problem() no longer rebuilds Q itself; this raw
                    # preprocess=False loop reads builder.Q directly, so it
                    # must rebuild here.
                    builder.build()
                except Exception as e:
                    self.logger.minimal(
                        f"Warning: could not decode window {builder.iter} "
                        f"({e}) — stopping early, the returned path is "
                        "incomplete and will fail validation"
                    )
                    break

            return {
                "solution": best_sample,
                "energy": best_energy,
                "optimized_params": self.params,
            }

        # preprocess=True: full pipeline with variable reduction and correction loop
        window_stats = []  # Track per-window variable reduction stats
        forced_collisions = []  # Track pre-processing forced collisions across all windows
        qpu_time_estimates = []  # Per-window pre-execution QPU time estimates (qiskit.remote only)
        correction_count = 0  # Track consecutive correction attempts for current window

        # Build the optimizer once — preserves internal state (e.g. Adam moments) across windows
        if self.optimizer_name == "GradientDescent":
            optimizer = qml.GradientDescentOptimizer()
        elif self.optimizer_name == "QNG":
            optimizer = qml.QNGOptimizer()
        elif self.optimizer_name == "SPSA":
            optimizer = qml.SPSAOptimizer()
        elif self.optimizer_name == "QNSPSA":
            optimizer = qml.QNSPSAOptimizer()
        else:
            optimizer = qml.AdamOptimizer()

        while (builder.total_t) > (builder.current_T):
            # TERMINATION CHECK: If all robots are inactive, stop solving
            active_robots = [r for r in builder.problem.robots.values() if r.active]
            if not active_robots:
                self.logger.standard(
                    "✅ All robots reached goal or inactive. Stopping solver."
                )
                break

            fixed_vars, window_stat, is_preprocessed, window_forced_collisions = (
                self._prepare_window(builder, bfs_variant, apply_numeric)
            )
            window_stats.append(window_stat)
            forced_collisions.extend(window_forced_collisions)

            # Track iteration time for quantum hardware
            if self.dev == "qiskit.remote" or self.dev == "qiskit.iqm":
                iteration_start = time.time()

            if is_preprocessed:
                self.logger.standard("⚡ Window fully pre-processed, skipping solver")
                full_sol, invalid_moves = self._handle_iteration_result(
                    {}, fixed_vars, builder
                )
                best_sample.append(full_sol)
                best_energy.append(0.0)
                continue

            if self.norm_scale != 0:
                builder.Q = self.normalize_qubo(builder.Q, self.norm_scale)

            # print("Start position:", builder.problem.start,
            #       "Iteration:", builder.iter)
            # Since pennylane doesn't inherently knows the index of the remember qubits, I need to pass them manually
            wires = builder.get_wires()
            num_qubits = len(wires)
            self.logger.standard(f"Number of qubits: {num_qubits}")

            # Determine if we need to remap wires for qiskit.remote
            if self.dev == "qiskit.remote" or self.dev == "qiskit.iqm":
                # Note that you need the final wires to be sequential, but that doesnt mean you need to sort the original
                # (Although it would make more sense there is no real reason to do that)
                # Diagrams are not even consistent so it doesn't really make a difference
                # Create mapping from original wire labels to SEQUENTIAL indices (0, 1, 2, ...)
                wire_remap = {orig_wire: idx for idx, orig_wire in enumerate(wires)}
                # print(f"Wire remap: {wire_remap}")
            else:
                wire_remap = None

            Hc, constant = builder.qubo_to_ising()

            # Remap Hamiltonian if using qiskit.remote
            if wire_remap is not None:
                # Remap the Hamiltonian to use sequential indices (0, 1, 2, ...)
                new_coeffs = []
                new_observables = []
                for coeff, obs in zip(Hc.coeffs, Hc.ops):
                    # Get the wires used in this observable
                    obs_wires = obs.wires
                    if len(obs_wires) == 1:
                        # Single qubit Pauli-Z - map to sequential index
                        new_wire = wire_remap[obs_wires[0]]
                        new_observables.append(qml.PauliZ(new_wire))
                    elif len(obs_wires) == 2:
                        # Two-qubit Pauli-Z tensor product - map both to sequential indices
                        new_wire1 = wire_remap[obs_wires[0]]
                        new_wire2 = wire_remap[obs_wires[1]]
                        new_observables.append(
                            qml.PauliZ(new_wire1) @ qml.PauliZ(new_wire2)
                        )
                    new_coeffs.append(coeff)
                Hc = qml.Hamiltonian(new_coeffs, new_observables)

                # Use sequential indices for mixer and circuit
                circuit_wires = range(num_qubits)
                Hmix = qml.qaoa.x_mixer(circuit_wires)
            else:
                Hmix = qml.qaoa.x_mixer(wires)
                circuit_wires = wires

            # Callable function to then be passed to qml.layer
            def qaoa_layer(gamma, beta):
                qml.qaoa.cost_layer(gamma, Hc)
                qml.qaoa.mixer_layer(beta, Hmix)

            if self.dev == "qiskit.remote":
                from quantum.hardware.ibm_device import IBMHardwareDevice

                # Only re-queries least_busy when num_qubits grows beyond current backend
                backend = self._get_backend(num_qubits)
                session = self._session_manager.get_session(
                    backend
                )  # None if unavailable/disabled

                # Use sequential indices for qiskit.remote (circuit_wires is already sorted)
                self.logger.standard("🔧 Initializing quantum device connection...")
                dev_start = time.time()
                # IBMHardwareDevice, not qml.device("qiskit.remote", ...): adds
                # job_id/usage telemetry and a pre-submission QPU-time estimate
                # around the stock QiskitDevice
                dev = IBMHardwareDevice(
                    wires=circuit_wires,
                    backend=backend,
                    session=session,
                )
                dev_time = time.time() - dev_start
                self.logger.standard(f"✓ Device initialized in {dev_time:.2f}s")
                self.logger.standard("=" * 60)
            elif self.dev == "qiskit.iqm":
                # No PennyLane device needed here: sampling goes through the Qrisp
                # bridge (_run_iqm_sampler), not a qiskit.remote qnode. dev is only
                # used below for cost_function, which optimization=True doesn't
                # support yet for this device (see the check further down).
                dev = None
            else:
                dev_start = time.time()
                dev = qml.device(self.dev, wires=circuit_wires)
                dev_time = time.time() - dev_start
                self.logger.debug(f"✓ Device initialized in {dev_time:.2f}s")
                self.logger.debug("=" * 60)

            # Create ansatz with the appropriate wire labels
            ansatz_circuit = self.create_ansatz(circuit_wires, qaoa_layer)

            shots = self.get_shots(num_qubits)
            self.logger.debug(f"Number of shots: {shots}")

            if self.dev != "qiskit.iqm":

                @qml.set_shots(shots)
                @qml.qnode(dev)
                # Note that this one is simply used for the optimization circuit
                def cost_function(params):
                    ansatz_circuit(params)
                    return qml.expval(Hc)

            # Optimization
            # (optimizer was built once before the loop to preserve internal state)
            if optimization:
                if self.dev == "qiskit.iqm":
                    raise NotImplementedError(
                        "optimization=True isn't supported yet for device="
                        "'qiskit.iqm' — only the Qrisp-bridge sampling path is "
                        "implemented. Pre-tune self.params offline, or use "
                        "optimization=False (the default)."
                    )
                # prev_cost = 0

                for step in range(self.optimizer_steps):
                    # Retrieving optimal parameters
                    step_start = time.time()
                    self.params, cost = optimizer.step_and_cost(
                        cost_function, self.params
                    )
                    step_time = time.time() - step_start

                    if step % 10 == 0:
                        if self.dev == "qiskit.remote":
                            self.logger.debug(
                                f"Step {step}, ⟨H_C⟩ = {cost:.6f}, Time: {step_time:.2f}s"
                            )
                        else:
                            self.logger.debug(f"Step {step}, ⟨H_C⟩ = {cost:.6f}")

            # Collect samples and calculate energies
            samples = []
            energies = []

            # Compute reverse wire mapping once per window (not once per shot)
            reverse_map = (
                {idx: orig_wire for orig_wire, idx in wire_remap.items()}
                if wire_remap is not None
                else None
            )

            if self.dev == "qiskit.iqm":
                # Bypasses PennyLane's qiskit.remote Sampler entirely — see
                # _get_backend / _run_iqm_sampler for why (upstream Qiskit bug).
                self.logger.standard("\n" + "=" * 60)
                self.logger.standard(
                    f"⏳ Collecting {shots} samples from IQM hardware (Qrisp bridge)..."
                )
                self.logger.standard("   Waiting for quantum job to complete...")
                self.logger.standard("=" * 60)
                sample_start = time.time()
                counts, iqm_timing = self._run_iqm_sampler(
                    ansatz_circuit, self.params, num_qubits, shots
                )
                sample_time = time.time() - sample_start
                self.logger.standard("=" * 60)
                self.logger.standard(f"✓ Samples collected in {sample_time:.2f}s")
                self.logger.standard("=" * 60)

                qpu_time_estimates.append(
                    {
                        "device": "qiskit.iqm",
                        "wall_clock_sec": sample_time,
                        "iqm_timing": iqm_timing,
                    }
                )

                # One evaluation per unique outcome is enough — best_idx below only
                # ever needs the best-energy outcome, not every individual shot.
                for bitstring, _count in counts.items():
                    # Qiskit/Qrisp bitstrings are little-endian: rightmost char = qubit 0
                    bits = bitstring[::-1]
                    binary_sample = {
                        wire: int(bits[i]) for i, wire in enumerate(circuit_wires)
                    }
                    if reverse_map is not None:
                        binary_sample = {
                            reverse_map[seq_idx]: value
                            for seq_idx, value in binary_sample.items()
                        }

                    samples.append(binary_sample)

                    energy = 0
                    for i in wires:
                        for j in wires:
                            if (i, j) in builder.Q:
                                energy += (
                                    builder.Q[(i, j)]
                                    * binary_sample[i]
                                    * binary_sample[j]
                                )
                    energies.append(energy)

            else:

                @qml.set_shots(shots)
                @qml.qnode(dev)
                def sample_circuit(params):
                    ansatz_circuit(params)
                    # Return measurements for all qubits
                    return qml.sample()

                # Get samples from the quantum circuit
                if self.dev == "qiskit.remote":
                    # dev (IBMHardwareDevice) logs its own pre-submission QPU-time
                    # estimate and job_id as part of sample_circuit() below
                    session_status = (
                        f"🔐 via Session on {backend.name}"
                        if session is not None
                        else "🌐 via public queue (no session)"
                    )
                    self.logger.standard("\n" + "=" * 60)
                    self.logger.standard(
                        f"⏳ Collecting {shots} samples from quantum hardware — {session_status}"
                    )
                    self.logger.standard("   Waiting for quantum job to complete...")
                    self.logger.standard("=" * 60)
                    sample_start = time.time()
                    raw_samples = sample_circuit(self.params)
                    sample_time = time.time() - sample_start
                    self.logger.standard("=" * 60)
                    self.logger.standard(f"✓ Samples collected in {sample_time:.2f}s")
                    self.logger.standard("=" * 60)

                    qpu_estimate_entry = {
                        "device": "qiskit.remote",
                        "wall_clock_sec": sample_time,
                        "gate_model": dev.last_gate_estimate,
                        "clops_model": dev.last_clops_estimate,
                    }
                    usage_info = dev.get_usage()
                    if usage_info:
                        self.logger.standard(
                            f"💰 Billed QPU usage: {usage_info['usage']}"
                        )
                        qpu_estimate_entry["billed_usage"] = usage_info
                    qpu_time_estimates.append(qpu_estimate_entry)
                else:
                    raw_samples = sample_circuit(self.params)

                # Handle different output formats
                if raw_samples.ndim == 1:
                    # Single shot case - convert to 2D
                    raw_samples = raw_samples.reshape(1, -1)

                # Process each sample
                for shot_idx in range(min(len(raw_samples), shots)):
                    sample_data = raw_samples[shot_idx]

                    # Convert from {-1, 1} to {0, 1} format
                    binary_sample = {}
                    for i, wire in enumerate(circuit_wires):
                        # Handle potential measurement outcomes
                        measurement = (
                            sample_data[i]
                            if hasattr(sample_data, "__getitem__")
                            else sample_data
                        )
                        # Convert Pauli-Z eigenvalues {-1, 1} to binary {0, 1}
                        binary_sample[wire] = int((measurement + 1) // 2)

                    # If using qiskit.remote, map sequential indices back to original wire labels
                    if reverse_map is not None:
                        binary_sample = {
                            reverse_map[seq_idx]: value
                            for seq_idx, value in binary_sample.items()
                        }

                    samples.append(binary_sample)

                    # Calculate QUBO energy for this sample
                    energy = 0
                    for i in wires:
                        for j in wires:
                            if (i, j) in builder.Q:
                                energy += (
                                    builder.Q[(i, j)]
                                    * binary_sample[i]
                                    * binary_sample[j]
                                )

                    energies.append(energy)

            self.logger.debug(
                f"Collected {len(samples)} samples with energies: {energies[:5]}..."
            )

            # Handle case where no samples were collected
            if not samples:
                self.logger.minimal(
                    "Warning: No samples collected, using random sample"
                )
                # Create a random binary sample as fallback
                random_sample = {i: np.random.randint(2) for i in wires}
                samples.append(random_sample)
                # Calculate energy for random sample
                energy = sum(
                    builder.Q.get((i, j), 0) * random_sample[i] * random_sample[j]
                    for i in wires
                    for j in wires
                )
                energies.append(energy)

            # Find best sample
            best_idx = np.argmin(energies)
            full_sol, invalid_moves = self._handle_iteration_result(
                samples[best_idx], fixed_vars, builder
            )
            best_sample.append(full_sol)
            best_energy.append(energies[best_idx])

            # Check if correction is needed due to invalid moves
            if invalid_moves:
                correction_count += 1
                self.logger.standard(
                    f"🔄 Correction attempt {correction_count}/{self.max_corrections} for current window"
                )

                if correction_count >= self.max_corrections:
                    self.logger.minimal(
                        f"⚠️  Max corrections ({self.max_corrections}) exceeded at t={builder.current_T}. "
                        f"Keeping last result (invalid moves for robots {list(invalid_moves.keys())})."
                    )

                    # CRITICAL: Force builder to advance to next window to avoid infinite loop
                    # Decode the path from the invalid solution and update the problem
                    path = self.decode_path(
                        full_sol, builder.problem, t_offset=builder.current_T
                    )
                    robot_paths = self.get_robot_paths(path)
                    robot_paths = self._resolve_duplicate_timesteps(
                        robot_paths, builder.problem
                    )
                    # Note: We skip _resolve_invalid_moves since we already know there are invalid moves
                    # and we want to accept them to move forward
                    builder.update_problem(robot_paths)

                    correction_count = 0
                # else: next loop iteration calls _prepare_window to rebuild from scratch
            else:
                correction_count = 0

            self.logger.debug(f"Best energy this iteration: {energies[best_idx]}")
            self.logger.debug(f"Best sample: {samples[best_idx]}")

            # Print iteration summary for quantum hardware
            if self.dev == "qiskit.remote" or self.dev == "qiskit.iqm":
                iteration_time = time.time() - iteration_start
                self.logger.standard("\n" + "=" * 60)
                self.logger.standard(f"✓ Iteration completed in {iteration_time:.2f}s")
                self.logger.standard(f"  Best energy: {energies[best_idx]:.6f}")
                self.logger.standard("=" * 60 + "\n")

        # Build final solution from stored robot paths (this is the correct solution)
        final_solution = self.build_solution_from_robot_paths(builder.problem)

        return {
            "solution": final_solution,  # Use solution built from robot paths
            "energy": best_energy,
            "optimized_params": self.params,
            "metadata": {
                "window_stats": window_stats,  # Per-window variable reduction stats
                "forced_collisions": forced_collisions,  # Pre-processing forced collisions (bypass K_crash/K_swap)
                "qpu_time_estimates": qpu_time_estimates,  # Per-window pre-execution QPU time estimates (qiskit.remote only)
            },
        }
