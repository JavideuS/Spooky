# Real-time status information during quantum computation
std_msgs/Header header
int32 current_iteration           # Current optimization iteration
float32 current_cost             # Current best cost found
float32 convergence_progress     # Progress towards convergence (0.0-1.0)
int32 quantum_shots_completed    # Quantum shots completed so far
string current_phase            # Current computation phase
float32 estimated_time_remaining # Estimated seconds remaining
