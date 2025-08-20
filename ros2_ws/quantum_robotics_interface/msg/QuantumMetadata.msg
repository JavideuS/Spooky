# Quantum metadata related to the path planning
float32 total_cost                 # Total path cost
int32 total_time_steps            # Total discrete time steps
string algorithm_used             # Algorithm that generated this path
geometry_msgs/Twist suggested_velocity  # Suggested velocity for this segment
builtin_interfaces/Duration expected_duration         # Expected time to reach this waypoint
float32 quantum_advantage_ratio   # Performance ratio vs classical
bool contains_superposition       # Whether path exploits quantum superposition
geometry_msgs/PoseStamped[] alternative_poses  # Alternative poses in superposition