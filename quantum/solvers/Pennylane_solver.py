import pennylane as qml
from pennylane import numpy as np
from .base_solver import BaseSolver


class PennylaneSolver(BaseSolver):
    def __init__(self, normalize_scale=0, num_reads=100, layers=2,
                 optimizer="GradientDescent", opt_steps=10,
                 device="default.qubit",
                 params=None, **kwargs):
        super().__init__(backend="pennylane", normalize_scale=normalize_scale,
                         num_reads=num_reads, **kwargs)
        self.optimizer_name = optimizer
        self.p = layers  # Number of QAOA layers
        self.dev = device
        # Parameters for the QAOA circuit
        self.params = (params if params is not None else
                       np.random.rand(layers, 2))
        self.optimizer_steps = opt_steps  # Number of optimization steps
        self.shots = num_reads  # Number of shots for sampling

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
        if device not in ["default.qubit", "lightning.qubit", "lightning.gpu"]:
            raise ValueError("Device must be either 'default.qubit' "
                             "or 'lightning'")
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
            # builder.build()
            if self.norm_scale != 0:
                builder.Q = self.normalize_qubo(builder.Q, self.norm_scale)

            # print("Start position:", builder.problem.start,
            #       "Iteration:", builder.iter)

            wires = range(builder.get_num_wires())
            print(f"Number of qubits: {len(wires)}")
            
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
                        print(f"Step {step}, ⟨H_C⟩ = {cost:.6f}")
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
                print("Warning: No samples collected, using random sample")
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
                print(f"Warning: Could not decode path: {e}")
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

        while (builder.total_t) > (builder.current_T):

            if self.norm_scale != 0:
                fixed_vars = builder.get_fixed_variables()
                builder.Q, offset = builder.reduce_qubo(fixed_vars)
                diag_fixed = builder.reduce_diag_fixed_vars_iterative()
                fixed_vars.update(diag_fixed)

                # In case the full qubo gets pre-processed
                if len(builder.Q) == 0:
                    print("Full QUBO gets pre-processed")
                    full_sol = self._handle_iteration_result(
                        {}, fixed_vars, builder)
                    best_sample.append(full_sol)
                    
                    continue
                builder.Q = self.normalize_qubo(builder.Q, self.norm_scale)

            # print("Start position:", builder.problem.start,
            #       "Iteration:", builder.iter)
            # Since pennylane doesn't inherently knows the index of the remember qubits, I need to pass them manually
            wires = builder.get_wires()
            print(f"Number of qubits: {len(wires)}")
            
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
                        print(f"Step {step}, ⟨H_C⟩ = {cost:.6f}")
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
                print("Warning: No samples collected, using random sample")
                # Create a random binary sample as fallback
                random_sample = {i: np.random.randint(2) for i in wires}
                samples.append(random_sample)
                # Calculate energy for random sample
                energy = sum(builder.Q.get((i, j), 0) * random_sample[i] * random_sample[j]
                           for i in wires for j in wires)
                energies.append(energy)

            # Find best sample
            best_idx = np.argmin(energies)
            full_sol = self._handle_iteration_result(samples[best_idx], fixed_vars, builder)
            best_sample.append(full_sol)
            best_energy.append(energies[best_idx])
            
            # print(f"Best energy this iteration: {energies[best_idx]}")
            # print(f"Best sample: {samples[best_idx]}")

        return {
            'solution': best_sample,
            'energy': best_energy,
            'optimized_params': self.params
        }
