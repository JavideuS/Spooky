# Quantum planner selection and configuration
string solver                  # Planner identifier ("qaoa", "vqe", "quantum_astar")
QuantumPlannerParameters params    # Quantum-specific parameters
bool fallback_to_classical        # Fall back to classical if quantum fails
string classical_backup           # Classical algorithm to use as backup ("dijkstra", "a_star")