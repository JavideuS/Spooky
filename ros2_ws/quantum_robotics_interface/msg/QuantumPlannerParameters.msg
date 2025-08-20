# Quantum-specific planner parameters
int32 num_layers                    # QAOA p parameter (number of layers)
int32 optimization_steps          # Maximum optimization iterations
float32 convergence_threshold      # Convergence criteria for optimizer
string backend                     # Qiskit backend ("qasm_simulator", "ibmq_qasm_simulator", etc.)
int32 shots                        # Number of quantum circuit shots
bool use_noise_model              # Whether to apply noise model
string optimizer                   # Classical optimizer ("COBYLA", "SPSA", etc.)
float32 timeout_seconds           # Maximum time allowed for quantum computation
bool enable_classical_comparison   # Whether to compare with classical pathfinding