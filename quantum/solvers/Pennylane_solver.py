import pennylane as qml
from pennylane import numpy as np
from .base_solver import BaseSolver
from qiskit_ibm_runtime import QiskitRuntimeService
import time

class PennylaneSolver(BaseSolver):
    def __init__(self, normalize_scale=0, num_reads="auto", layers=2,
                 optimizer="GradientDescent", opt_steps=10,
                 device="default.qubit",
                 params=None, verbose_level=2, **kwargs):
        super().__init__(solver="pennylane", normalize_scale=normalize_scale,
                         num_reads=num_reads, verbose_level=verbose_level, **kwargs)
        self.optimizer_name = optimizer
        self.p = layers  # Number of QAOA layers
        self.dev = device
        if device == "qiskit.remote":
            self.service = QiskitRuntimeService()
            # self.backend = self.service.least_busy(operational=True, simulator=False, min_num_qubits=100)
        # Parameters for the QAOA circuit
        self.params = (params if params is not None else
                       np.random.rand(layers, 2))
        self.optimizer_steps = opt_steps  # Number of optimization steps
        self.shots = num_reads  # Number of shots for sampling

    def get_shots(self, num_qubits):
        if self.shots == "auto":
            if num_qubits <= 9:
                return 500
            elif num_qubits <= 12:
                return 2500
            elif num_qubits <= 16:
                return 10000 #15000 safer
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
        if optimizer not in ["GradientDescent", "Adam", "QNG", "SPSA", "QNSPSA"]:
            raise ValueError("Optimizer must be either 'GradientDescent', "
                             "'QNG', 'SPSA', 'QNSPSA' or 'Adam'")
        if device not in ["default.qubit", "lightning.qubit", "lightning.gpu", "qiskit.remote"]:
            raise ValueError("Device must be either 'default.qubit' "
                             "or 'lightning' or 'qiskit.remote'")
        return cls(normalize_scale=norm_scale, num_reads=num_reads,
                   layers=layers, optimizer=optimizer, device=device,
                   params=params, opt_steps=opt_steps)

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

    def solve_qubo(self, builder, optimization=True):
        """
        Solve QUBO using Pennylane QAOA.

        Args:
            builder: QUBOBuilder instance

        Returns:
            Dictionary containing solution, energy, and raw response
        """
        best_sample = []
        best_energy = []

        while (builder.total_t) > (builder.current_T):
            if self.norm_scale != 0:
                builder.Q = self.normalize_qubo(builder.Q, self.norm_scale)

            wires = range(builder.get_num_wires())
            self.logger.standard(f"Number of qubits: {len(wires)}")
            
            Hc, constant = builder.qubo_to_ising()
            Hmix = qml.qaoa.x_mixer(wires)

            # Callable function to then be passed to qml.layer
            def qaoa_layer(gamma, beta):
                qml.qaoa.cost_layer(gamma, Hc)
                qml.qaoa.mixer_layer(beta, Hmix)

            ansatz_circuit = self.create_ansatz(wires, qaoa_layer)

            dev = qml.device(self.dev, wires=wires, shots=self.shots)

            @qml.qnode(dev)
            def cost_function(params):
                ansatz_circuit(params)
                return qml.expval(Hc)

            # Optimization
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

            if optimization:
                # prev_cost = 0
                for step in range(self.optimizer_steps):
                    # Retrieving optimal parameters
                    self.params, cost = optimizer.step_and_cost(
                        cost_function, self.params)
                    if step % 10 == 0:
                        self.logger.debug(f"Step {step}, ⟨H_C⟩ = {cost:.6f}")
                    # if step > 3 and abs(cost - prev_cost) < 1e-4:
                    #     break
                    # prev_cost = cost

            @qml.qnode(dev)
            def sample_circuit(params):
                ansatz_circuit(params)
                # Return measurements for all qubits
                return qml.sample()

            # print(f"Collecting {self.shots} samples...")
            
            # Collect samples and calculate energies
            samples = []
            energies = []
            
            # Get samples from the quantum circuit
            raw_samples = sample_circuit(self.params)
            
            # Handle different output formats
            if raw_samples.ndim == 1:
                # Single shot case - convert to 2D
                raw_samples = raw_samples.reshape(1, -1)
            
            # Process each sample
            for shot_idx in range(min(len(raw_samples), self.shots)):
                sample_data = raw_samples[shot_idx]
                
                # Convert from {-1, 1} to {0, 1} format
                binary_sample = {}
                for i, wire in enumerate(wires):
                    # Handle potential measurement outcomes
                    measurement = sample_data[i] if hasattr(sample_data, '__getitem__') else sample_data
                    # Convert Pauli-Z eigenvalues {-1, 1} to binary {0, 1}
                    binary_sample[wire] = int((measurement + 1) // 2)
                
                samples.append(binary_sample)
                
                # Calculate QUBO energy for this sample
                energy = 0
                for i in wires:
                    for j in wires:
                        if (i, j) in builder.Q:
                            energy += builder.Q[(i, j)] * binary_sample[i] * binary_sample[j]
                
                energies.append(energy)
                
            # print(f"Collected {len(samples)} samples with energies: {energies[:5]}...")
            
            # Handle case where no samples were collected
            if not samples:
                self.logger.minimal("Warning: No samples collected, using random sample")
                # Create a random binary sample as fallback
                random_sample = {i: np.random.randint(2) for i in wires}
                samples.append(random_sample)
                # Calculate energy for random sample
                energy = sum(builder.Q.get((i, j), 0) * random_sample[i] * random_sample[j]
                           for i in wires for j in wires)
                energies.append(energy)

            # Find best sample
            best_idx = np.argmin(energies)
            best_sample.append(samples[best_idx])
            best_energy.append(energies[best_idx])
            
            # print(f"Best energy this iteration: {energies[best_idx]}")
            # print(f"Best sample: {samples[best_idx]}")

            # Update problem for next iteration
            try:
                last_pos = self.decode_path(samples[best_idx], builder.problem)[-1]
                builder.update_problem(last_pos[:2])
            except Exception as e:
                self.logger.minimal(f"Warning: Could not decode path: {e}")
                # Handle gracefully or break the loop
                break

        return {
            'solution': best_sample,
            'energy': best_energy,
            'optimized_params': self.params
        }

    def solve_qubo_smart(self, builder, optimization=False):
        """
        Solve QUBO using Pennylane QAOA.

        Args:
            builder: QUBOBuilder instance

        Returns:
            Dictionary containing solution, energy, and raw response
        """
        best_sample = []
        best_energy = []
        window_stats = []  # Track per-window variable reduction stats
        correction_count = 0  # Track consecutive correction attempts for current window

        while (builder.total_t) > (builder.current_T):


            # TERMINATION CHECK: If all robots are inactive, stop solving
            active_robots = [r for r in builder.problem.robots.values() if r.active]
            if not active_robots:
                self.logger.standard("✅ All robots reached goal or inactive. Stopping solver.")
                break

            # Ensure QUBO is built for the start of the window
            # If the user forgot to call builder.build(), we do it here
            if not builder.Q:
                self.logger.debug("QUBO has not been built, calling builder.build()")
                builder.build()

            # Track iteration time for quantum hardware
            if self.dev == "qiskit.remote":
                iteration_start = time.time()


            if self.norm_scale != 0:
                # Get initial variable count BEFORE reduction
                initial_vars = builder.get_num_wires()
                
                fixed_vars = builder.get_fixed_variables()
                # Don't log initial reduction - only needed for BFS recalc which is rare
                builder.Q, offset, _ = builder.reduce_qubo(fixed_vars, log_reductions=False)
                diag_fixed = builder.reduce_diag_fixed_vars_iterative(initial_reduction_log=None)
                fixed_vars.update(diag_fixed)
                
                # Count total reduced variables (BFS fixed + diagonal fixed)
                vars_reduced = len(fixed_vars)
                
                # Get final variable count AFTER reduction
                final_vars = builder.get_num_wires()
                reduction_ratio = vars_reduced / initial_vars if initial_vars > 0 else 0
                
                # Store per-window stats
                window_stats.append({
                    "window": builder.iter,
                    "initial_variables": initial_vars,
                    "variables_reduced": vars_reduced,
                    "final_variables": final_vars,
                    "reduction_ratio": round(reduction_ratio, 4)
                })
                
                self.logger.standard(
                    f"Window {builder.iter}: {initial_vars} → {final_vars} vars "
                    f"(reduced {vars_reduced}, {reduction_ratio:.1%})"
                )

                # In case the full qubo gets pre-processed
                if len(builder.Q) == 0:
                    self.logger.standard("Full QUBO gets pre-processed", builder.current_T)
                    full_sol, invalid_moves = self._handle_iteration_result(
                        {}, fixed_vars, builder)
                    best_sample.append(full_sol)
                    
                    continue
                # print(fixed_vars)
                builder.Q = self.normalize_qubo(builder.Q, self.norm_scale)

            # print("Start position:", builder.problem.start,
            #       "Iteration:", builder.iter)
            # Since pennylane doesn't inherently knows the index of the remember qubits, I need to pass them manually
            wires = builder.get_wires()
            num_qubits = len(wires)
            self.logger.standard(f"Number of qubits: {num_qubits}")
            
            # Determine if we need to remap wires for qiskit.remote
            if self.dev == "qiskit.remote":
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
                        new_observables.append(qml.PauliZ(new_wire1) @ qml.PauliZ(new_wire2))
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
                
                self.logger.standard("=" * 60)
                self.logger.standard("🔍 Searching for available quantum hardware...")
                backend_start = time.time()
                backend = self.service.least_busy(operational=True, simulator=False, min_num_qubits=num_qubits)
                backend_time = time.time() - backend_start
                self.logger.standard(f"✓ Backend selected: {backend.name}")
                self.logger.standard(f"  Backend selection time: {backend_time:.2f}s")
                self.logger.standard("=" * 60)
    
                # Use sequential indices for qiskit.remote (circuit_wires is already sorted)
                self.logger.standard("🔧 Initializing quantum device connection...")
                dev_start = time.time()
                dev = qml.device('qiskit.remote', wires=circuit_wires, backend=backend)
                dev_time = time.time() - dev_start
                self.logger.standard(f"✓ Device initialized in {dev_time:.2f}s")
                self.logger.standard("=" * 60)
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
            @qml.set_shots(shots)
            @qml.qnode(dev)
            # Note that this one is simply used for the optimization circuit
            def cost_function(params):
                ansatz_circuit(params)
                return qml.expval(Hc)

            # Optimization
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

            if optimization:
                # prev_cost = 0

                for step in range(self.optimizer_steps):
                    # Retrieving optimal parameters
                    step_start = time.time()
                    self.params, cost = optimizer.step_and_cost(
                        cost_function, self.params)
                    step_time = time.time() - step_start
                    
                    if step % 10 == 0:
                        if self.dev == "qiskit.remote":
                            self.logger.debug(f"Step {step}, ⟨H_C⟩ = {cost:.6f}, Time: {step_time:.2f}s")
                        else:
                            self.logger.debug(f"Step {step}, ⟨H_C⟩ = {cost:.6f}")
                    # if step > 3 and abs(cost - prev_cost) < 1e-4:
                    #     break
                    # prev_cost = cost

            @qml.set_shots(shots)
            @qml.qnode(dev)
            def sample_circuit(params):
                ansatz_circuit(params)
                # Return measurements for all qubits
                return qml.sample()

            # print(f"Collecting {self.shots} samples...")
            
            # Collect samples and calculate energies
            samples = []
            energies = []
            
            # Get samples from the quantum circuit
            if self.dev == "qiskit.remote":
                self.logger.standard("\n" + "=" * 60)
                self.logger.standard(f"⏳ Collecting {shots} samples from quantum hardware...")
                self.logger.standard("   Waiting for quantum job to complete...")
                self.logger.standard("=" * 60)
                sample_start = time.time()
                raw_samples = sample_circuit(self.params)
                sample_time = time.time() - sample_start
                self.logger.standard("=" * 60)
                self.logger.standard(f"✓ Samples collected in {sample_time:.2f}s")
                self.logger.standard("=" * 60)
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
                    measurement = sample_data[i] if hasattr(sample_data, '__getitem__') else sample_data
                    # Convert Pauli-Z eigenvalues {-1, 1} to binary {0, 1}
                    binary_sample[wire] = int((measurement + 1) // 2)
                
                # If using qiskit.remote, map sequential indices back to original wire labels
                if wire_remap is not None:
                    # Create reverse mapping: sequential index -> original wire label
                    reverse_map = {idx: orig_wire for orig_wire, idx in wire_remap.items()}
                    # Remap the binary sample to use original wire labels
                    binary_sample = {reverse_map[seq_idx]: value for seq_idx, value in binary_sample.items()}
                
                samples.append(binary_sample)
                
                # Calculate QUBO energy for this sample
                energy = 0
                for i in wires:
                    for j in wires:
                        if (i, j) in builder.Q:
                            energy += builder.Q[(i, j)] * binary_sample[i] * binary_sample[j]
                
                energies.append(energy)
                
            # print(f"Collected {len(samples)} samples with energies: {energies[:5]}...")
            
            # Handle case where no samples were collected
            if not samples:
                self.logger.minimal("Warning: No samples collected, using random sample")
                # Create a random binary sample as fallback
                random_sample = {i: np.random.randint(2) for i in wires}
                samples.append(random_sample)
                # Calculate energy for random sample
                energy = sum(builder.Q.get((i, j), 0) * random_sample[i] * random_sample[j]
                           for i in wires for j in wires)
                energies.append(energy)

            # Find best sample
            best_idx = np.argmin(energies)
            full_sol, invalid_moves = self._handle_iteration_result(samples[best_idx], fixed_vars, builder)
            best_sample.append(full_sol)
            best_energy.append(energies[best_idx])
            
            # Check if correction is needed due to invalid moves
            if invalid_moves:
                correction_count += 1
                self.logger.standard(f"🔄 Correction attempt {correction_count}/{self.max_corrections} for current window")
                
                if correction_count >= self.max_corrections:
                    self.logger.minimal(f"⚠️  Max corrections ({self.max_corrections}) exceeded at t={builder.current_T}. "
                                       f"Keeping last result (invalid moves for robots {list(invalid_moves.keys())}).")
                    
                    # CRITICAL: Force builder to advance to next window to avoid infinite loop
                    # Decode the path from the invalid solution and update the problem
                    path = self.decode_path(full_sol, builder.problem, t_offset=builder.current_T)
                    robot_paths = self.get_robot_paths(path)
                    robot_paths = self._resolve_duplicate_timesteps(robot_paths, builder.problem)
                    # Note: We skip _resolve_invalid_moves since we already know there are invalid moves
                    # and we want to accept them to move forward
                    builder.update_problem(robot_paths)
                    
                    correction_count = 0  # Reset for next window
            else:
                # Successful window - reset correction counter
                correction_count = 0
            
            # print(f"Best energy this iteration: {energies[best_idx]}")
            # print(f"Best sample: {samples[best_idx]}")
            
            # Print iteration summary for quantum hardware
            if self.dev == "qiskit.remote":
                iteration_time = time.time() - iteration_start
                self.logger.standard("\n" + "=" * 60)
                self.logger.standard(f"✓ Iteration completed in {iteration_time:.2f}s")
                self.logger.standard(f"  Best energy: {energies[best_idx]:.6f}")
                self.logger.standard("=" * 60 + "\n")

        # Build final solution from stored robot paths (this is the correct solution)
        final_solution = self.build_solution_from_robot_paths(builder.problem)
        
        return {
            'solution': final_solution,  # Use solution built from robot paths
            'energy': best_energy,
            'optimized_params': self.params,
            'metadata': {
                'window_stats': window_stats,  # Per-window variable reduction stats
                # 'window_solutions': best_sample  # Keep window solutions for debugging
            }
        }
