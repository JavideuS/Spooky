import pennylane as qml
import numpy as np
from abc import ABC, abstractmethod
from quantum.utils import paths


class BaseQUBO(ABC):
    """
    Base class for QUBO builders that provides shared fields and common
    utilities. Subclasses should implement build() and may override any
    inherited methods as needed.

    Required attributes on the provided problem instance:
    - grid or graph structure as required by the subclass
    - start, end (optional depending on subclass)
    - T (total time horizon if using windowed approach)
    """

    def __init__(
        self,
        problem,
        penalties,
        name="unnamed",
        var_limit=601,
        window_max_steps=None,
        distance_scaling=None,
    ):
        self.problem = problem
        self.penalties = penalties
        self.name = name
        self.var_limit = var_limit
        # Windowing/time slicing support
        self.current_T = 0  # This represents the current global time in the problem
        self.total_t = getattr(self.problem, "T", 1)
        self.t_max = window_max_steps or self.max_window_size()
        self.iter = 0
        # QUBO dict
        self.Q = {}
        self.initial_num_vars = 0  # To be set by subclass during build
        # Optional knob used by grid subclass
        self.distance_scaling = distance_scaling
        self.prev_solution = []

    # Subclasses must implement build to populate self.Q
    @abstractmethod
    def build(self, constraints_to_apply=None):
        """Build the QUBO dictionary for the current window and return it."""
        raise NotImplementedError

    # Shared: QUBO -> Ising mapping (identical across formats)
    def qubo_to_ising(self):
        """
        Convert the QUBO (upper triangle) dictionary to an Ising Hamiltonian.
        Returns (qml.Hamiltonian, constant_offset).
        """
        linear_coeffs = {}
        quadratic_terms = {}
        constant = 0.0
        for (i, j), qij in self.Q.items():
            if i == j:
                constant += qij / 2
                linear_coeffs[i] = linear_coeffs.get(i, 0) - qij / 2
            else:
                constant += qij / 4
                linear_coeffs[i] = linear_coeffs.get(i, 0) - qij / 4
                linear_coeffs[j] = linear_coeffs.get(j, 0) - qij / 4
                quadratic_terms[i, j] = (
                    quadratic_terms.get((i, j), 0) + qij / 4
                )

        coeffs = []
        observables = []
        for i in sorted(linear_coeffs):
            if linear_coeffs[i] != 0:
                coeffs.append(linear_coeffs[i])
                observables.append(qml.PauliZ(i))
        for (i, j), val in quadratic_terms.items():
            if val != 0:
                coeffs.append(val)
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
        Hc = qml.Hamiltonian(coeffs, observables)
        return Hc, constant

    def get_wires(self):
        """
        Return set of unique variable indices in the current QUBO.
        """
        if not self.Q:
            raise ValueError("QUBO dictionary is empty. Build the QUBO first.")
        qubit_indices = set()
        for (i, j) in self.Q.keys():
            qubit_indices.update([i, j])
        return qubit_indices

    # Shared: count wires from Q
    def get_num_wires(self):
        """
        Return the number of unique variable indices in the current QUBO.
        """
        return len(self.get_wires())

    # Shared: compute max window size based on var_limit
    def max_window_size(self):
        """
        Estimate max window size using problem dimensions if available.
        Subclasses relying on non-grid encodings can override this.
        """
        if self.problem.get_format_type() == "graph":
            vars_per_time = self.num_nodes
        elif self.problem.get_format_type() == "grid":
            M = self.problem.grid.M
            N = self.problem.grid.N
            vars_per_time = M * N
        
        robot_active_timeline = self.problem.get_robot_per_timestep()
        needed_vars = 0
        for t in range(self.current_T, self.total_t):
            num_active_robots = len(robot_active_timeline[t])
            needed_vars += num_active_robots * vars_per_time
            if needed_vars > self.var_limit:
                return t - self.current_T
        # We need to substract initial offset from current_T
        # The +1 is because range is exclusive at the end (else it would keep infinite loop)
        return self.total_t - self.current_T + 1

    def get_active_robot_in_window(self):
        """
        Get list of robot IDs that are active in the current window.
        They are sorted by priority in descending order.
        """
        active_robots = set()
        robot_active_timeline = self.problem.get_robot_per_timestep()
        for t in range(self.current_T, self.current_T + self.t_max):
            for robot_id in robot_active_timeline.get(t, []):
                active_robots.add(robot_id)
        return sorted(active_robots, key=lambda x: self.problem.robots[x].priority, reverse=True)
    
    def get_active_robots_per_timestep_in_window(self):
        """
        Get a dict mapping each timestep in the current window
        to the list of active robot IDs at that timestep.
        """
        robot_active_timeline = self.problem.get_robot_per_timestep()
        active_per_timestep = {}

        for t in range(self.current_T, self.current_T + self.t_max):
            active_robots = robot_active_timeline.get(t, [])
            if active_robots:  # only include if something active
                active_per_timestep[t] = active_robots

        return active_per_timestep

    # Shared: window update/reset
    def update_problem(self, solution=[]):
        """Advance window and optionally update start state for next build."""
        self.iter += 1  # Now iter is only used for information
        # The new method to keep time is current_T which to keep track of overlapping first step with last step
        # It simply does t_max - 1 (meaning that if you only can render one time step you will always be in that time)
        # Because you need to render the step where you were and the following iteration (qubo renders the transition)
        # Note that this also helps debug since now the global time don't care about iterations (which clipped time)
        self.current_T += self.t_max - 1
        # We update current time and recalculate t_max for next window
        self.t_max = self.max_window_size()
        if self.t_max > 0:
            for robot_num, new_segment in solution.items():
                robot_id = list(self.problem.robots.keys())[robot_num]
                old_path = self.problem.robots[robot_id].path
                merged = paths.merge_paths(old_path, new_segment)

                self.problem.robots[robot_id].path = merged
                self.problem.robots[robot_id].current_position = merged[-1][:2]
            self.build()

    def reset_problem(self):
        """Reset windowing and restore initial start position if available."""
        for robot_id in self.problem.robots.keys():
            robot = self.problem.robots[robot_id]
            robot.path = []
            robot.current_position = robot.start
        self.iter = 0
        self.current_T = 0
        self.t_max = self.max_window_size()

    # Shared utilities for Q manipulation
    def dict_to_array(self, fill_value=0):
        if not self.Q:
            return np.array([[]])
        rows, cols = zip(*self.Q.keys())
        shape = (max(rows) + 1, max(cols) + 1)
        arr = np.full(shape, fill_value, dtype=float)
        for (r, c), val in self.Q.items():
            arr[r, c] = val
        return arr

    def reduce_qubo(self, fixed_vars):
        """
        Reduce a QUBO dictionary by fixing variables.
        Assumes Q uses upper triangular form: (i,j) with i <= j only.
        
        Args:
            Q: QUBO dictionary
            fixed_vars: dict {idx: value} or numpy array
        
        Returns:
            (reduced_Q, const_offset, reduction_log)
        """
        if isinstance(fixed_vars, np.ndarray):
            fixed_dict = {
                i: int(v) for i, v in enumerate(fixed_vars) if not np.isnan(v)
            }
        else:
            fixed_dict = fixed_vars
        # print("fixed_dict", fixed_dict)
        reduced_Q = {}
        const_offset = 0
        reduction_log = []  # Track all changes
        
        for key, coeff in self.Q.items():
            i, j = key
            i_fixed = i in fixed_dict
            j_fixed = j in fixed_dict
            
            if not i_fixed and not j_fixed:
                # Both free: keep as is
                reduced_Q[key] = coeff
            elif i_fixed and j_fixed:
                # Both fixed: add to constant
                const_offset += coeff * fixed_dict[i] * fixed_dict[j]
                reduction_log.append({
                    'type': 'both_fixed',
                    'original_key': key,
                    'coeff': coeff,
                    'fixed_values': (fixed_dict[i], fixed_dict[j])
                })
            elif i_fixed:
                # Only i fixed
                if fixed_dict[i] == 1:
                    reduced_Q[(j, j)] = reduced_Q.get((j, j), 0) + coeff
                    reduction_log.append({
                        'type': 'i_fixed',
                        'original_key': key,
                        'coeff': coeff,
                        'fixed_var': i,
                        'fixed_value': fixed_dict[i],
                        'free_var': j
                    })
            else:  # j_fixed
                # Only j fixed
                if fixed_dict[j] == 1:
                    reduced_Q[(i, i)] = reduced_Q.get((i, i), 0) + coeff
                    reduction_log.append({
                        'type': 'j_fixed',
                        'original_key': key,
                        'coeff': coeff,
                        'fixed_var': j,
                        'fixed_value': fixed_dict[j],
                        'free_var': i
                    })
        
        return reduced_Q, const_offset, reduction_log
    
    def reverse_reduction(self, reduced_Q, reduction_log, var_to_unfix):
        """
        Reverse the reduction for a specific variable.
        
        Args:
            reduced_Q: The reduced QUBO dictionary
            reduction_log: Log from reduce_qubo
            var_to_unfix: Variable index to unfix
        
        Returns:
            (restored_Q, removed_const_offset)
        """
        restored_Q = reduced_Q.copy()
        removed_const = 0
        
        for entry in reduction_log:
            orig_key = entry['original_key']
            i, j = orig_key
            coeff = entry['coeff']
            
            if entry['type'] == 'both_fixed':
                # Check if either variable is the one we're unfixing
                if i == var_to_unfix or j == var_to_unfix:
                    # Remove from constant
                    fixed_vals = entry['fixed_values']
                    removed_const += coeff * fixed_vals[0] * fixed_vals[1]
                    
                    # If only one is being unfixed, convert to linear term
                    if i == var_to_unfix and j != var_to_unfix:
                        # j is still fixed
                        if fixed_vals[1] == 1:
                            restored_Q[(i, i)] = restored_Q.get((i, i), 0) + coeff
                    elif j == var_to_unfix and i != var_to_unfix:
                        # i is still fixed
                        if fixed_vals[0] == 1:
                            restored_Q[(j, j)] = restored_Q.get((j, j), 0) + coeff
                    else:
                        # Both being unfixed, restore original term
                        restored_Q[orig_key] = coeff
            
            elif entry['type'] == 'i_fixed':
                # i was fixed, j was free
                if entry['fixed_var'] == var_to_unfix:
                    free_var = entry['free_var']
                    # Remove the linear term we added
                    if entry['fixed_value'] == 1:
                        restored_Q[(free_var, free_var)] -= coeff
                        if abs(restored_Q[(free_var, free_var)]) < 1e-10:
                            del restored_Q[(free_var, free_var)]
                    # Restore original interaction term
                    restored_Q[orig_key] = coeff
            
            elif entry['type'] == 'j_fixed':
                # j was fixed, i was free
                if entry['fixed_var'] == var_to_unfix:
                    free_var = entry['free_var']
                    # Remove the linear term we added
                    if entry['fixed_value'] == 1:
                        restored_Q[(free_var, free_var)] -= coeff
                        if abs(restored_Q[(free_var, free_var)]) < 1e-10:
                            del restored_Q[(free_var, free_var)]
                    # Restore original interaction term
                    restored_Q[orig_key] = coeff
        
        return restored_Q, removed_const

    def diag_fixed_vars(self):
        """
        Identify fixed variables based on outliers in diagonal coefficients
        per robot and time step.

        For each robot and time step:
        1. Collect all diagonal coefficients for variables in that time step for that robot
        2. If there are multiple variables with different coefficients:
           - Variables with smaller coefficients are fixed to 0 (outliers)
           - If one variable has a significantly larger coefficient,
             it's fixed to 1
        3. If there's only one variable in a time step, it's fixed to 1
        
        The key difference from the previous implementation is that this processes
        robots sequentially, then timesteps within each robot, reducing the QUBO
        after EACH TIMESTEP. This prevents:
        - Fixing variables at the same timestep for different robots simultaneously
        - Fixing non-adjacent positions across consecutive timesteps for the same robot
        Both of which could create crash scenarios or invalid paths without knowing.

        Note that this functions works better when there was already a
        pre-reduction. And be mind that it is iterative, when it clears at
        one iteration, it updates coefficient so it can find more fixed
        variables in the next iteration.

        Returns:
            dict: {variable_index: fixed_value} where fixed_value is 0 or 1
        """
        from collections import Counter
        total_fixed = {}  # Track all fixed variables across all robots
        # Use instance variable self.reduction_log to track ALL reductions
        # This is initialized in reduce_diag_fixed_vars_iterative
        # We could get all dictionary items, but it simpler to iterate
        # over known qubit numbers
        n = self.initial_num_vars
        type = self.problem.get_format_type()
        if type == "grid":
            M, N = self.problem.grid.M, self.problem.grid.N
            vars_per_time = M * N
            adjacency_dict = self.problem.grid.adjacency
        elif type == "graph":
            vars_per_time = self.num_nodes
            adjacency_dict = self.problem.graph.adjacency
        
        # Get robot information
        robot_nums = self.problem.get_robot_nums()
        num_robots = self.problem.num_robots

        # Helper function to check adjacency
        def is_adjacent(pos1, pos2):
            """Check if pos2 is adjacent to pos1"""
            if type == "grid":
                # For grid: pos1 and pos2 are (i, j) tuples
                return pos2 in adjacency_dict.get(pos1, [])
            else:  # graph
                # For graph: pos1 and pos2 are (i, j) tuples from decode_position
                # Convert to node IDs using graph's get_node_from_position
                node1 = self.problem.graph.get_node_from_position(pos1)
                node2 = self.problem.graph.get_node_from_position(pos2)
                if node1 is None or node2 is None:
                    return False
                # adjacency_dict[node] returns list of (neighbor, weight) tuples
                neighbors = [neighbor for neighbor, weight in adjacency_dict.get(node1, [])]
                return node2 in neighbors
        
        # Process per robot, then per timestep within each robot
        for robot_id in self.get_active_robot_in_window():
            robot_num = robot_nums[robot_id]
            robot_offset = robot_num * (vars_per_time * self.total_t)
            
            # Collect variables for this robot's timesteps
            robot_time_step_vars = {}

            prev_fixed_pos = None
            prev_timestep = None
            
            for i in range(n):
                if (i, i) in self.Q:
                    diag_coeff = self.Q[(i, i)]
                    if diag_coeff != 0:
                        # Check if this variable belongs to current robot
                        if i >= robot_offset and i < robot_offset + (vars_per_time * self.total_t):
                            # Calculate timestep within this robot's variables
                            local_idx = i - robot_offset
                            t = local_idx // vars_per_time
                            if t not in robot_time_step_vars:
                                robot_time_step_vars[t] = []
                            robot_time_step_vars[t].append((i, diag_coeff))
            
            # Process each timestep for this robot, reducing after each timestep
            # Sort timesteps to process them in order
            for t in sorted(robot_time_step_vars.keys()):
                timestep_fixed = {}  # Fixed variables for this timestep only
                bfs_adjusted = False  # Flag to track if we did BFS adjustment
                
                should_validate_adjacency = (
                    prev_fixed_pos is not None and 
                    prev_timestep is not None and 
                    t == prev_timestep + 1
                )
                
                # print(f"Robot {robot_id}, Time step {t}: vars and diag coeffs: {robot_time_step_vars[t]}")
                # If only one variable, fix to 1
                if len(robot_time_step_vars[t]) == 1:
                    var_idx = robot_time_step_vars[t][0][0]
                    timestep_fixed[var_idx] = 1

                    prev_fixed_pos = var_idx
                    prev_timestep = t
                
                # To validate staying in goal
                goal = self.problem.robots[robot_id].goal
                counts = Counter([v[1] for v in robot_time_step_vars[t]])
                # If all coefficients are the same, skip
                if len(counts) <= 1:
                    continue
                else:
                    # Since we are minimizing, the minimum is the best (set to 1)
                    min_coeff = min(counts)
                    if counts[min_coeff] == 1:
                        for var_idx, coeff in robot_time_step_vars[t]:
                            if coeff == min_coeff:
                                if should_validate_adjacency:
                                    i, j, t_decoded, robot_num = paths.decode_position(var_idx, self.problem)
                                    prev_i, prev_j, prev_t, prev_robot_num = paths.decode_position(prev_fixed_pos, self.problem)
                                    # print(prev_i, prev_j, prev_t, prev_robot_num, var_idx)
                                    # print(i, j, t_decoded, robot_num, prev_fixed_pos)
                                    # Check if we can move FROM (prev_i, prev_j) TO (i, j)
                                    print(prev_i, prev_j, i, j, is_adjacent((prev_i, prev_j), (i, j)))
                                    if not is_adjacent((prev_i, prev_j), (i, j)) and (i, j) != goal: 
                                        print(var_idx, f"not fixed to 1 for robot {robot_id}, time step {t}")
                                        print("Recalculating QUBO BFS")
                                        # Calculate reachable positions from the previous fixed position
                                        
                                        # Determine start position format based on problem type
                                        if type == "grid":
                                            start_pos = (prev_i, prev_j)
                                        else:  # graph
                                            # For graph, we need to use the node ID
                                            start_pos = self.problem.graph.get_node_from_position((prev_i, prev_j))
                                        
                                        reachable = self.reachable_positions_aggressive(
                                            self.problem.robots[robot_id], 
                                            start_pos, 
                                            prev_timestep, 
                                            len(robot_time_step_vars) + 1
                                        )
                                        print("New reach", reachable)
                                        
                                        # Process each timestep from current t onwards
                                        # Use reachable.keys() instead of robot_time_step_vars.keys() because
                                        # some timesteps might not have had diagonal coefficients initially
                                        for t_check in sorted([k for k in reachable.keys() if k >= t]):
                                            reachable_positions = reachable[t_check]
                                            print(f"Time {t_check}: reachable positions:", reachable_positions)
                                            
                                            # Get variables for this timestep if they exist
                                            timestep_vars = robot_time_step_vars.get(t_check, [])
                                            print(f"Time {t_check}: variables:", timestep_vars)
                                            
                                            # Decode each variable to get its position
                                            vars_to_fix_zero = []
                                            vars_to_unfix = []
                                            reachable_vars = []  # Track reachable variables for this timestep
                                            
                                            for var_idx_check, coeff_check in timestep_vars:
                                                i_check, j_check, t_decoded, robot_num_check = paths.decode_position(var_idx_check, self.problem)
                                                
                                                # Determine position format based on problem type
                                                if type == "grid":
                                                    pos_check = (i_check, j_check)
                                                else:  # graph
                                                    pos_check = self.problem.graph.get_node_from_position((i_check, j_check))
                                                
                                                # Check if this position is reachable
                                                if pos_check not in reachable_positions:
                                                    # Position not reachable, should be fixed to 0
                                                    if var_idx_check not in total_fixed or total_fixed[var_idx_check] != 0:
                                                        vars_to_fix_zero.append(var_idx_check)
                                                        print(f"  Variable {var_idx_check} at position {pos_check} NOT reachable, fixing to 0")
                                                else:
                                                    # Position is reachable
                                                    # If it was previously fixed to 0, we need to unfix it
                                                    if var_idx_check in total_fixed and total_fixed[var_idx_check] == 0:
                                                        vars_to_unfix.append(var_idx_check)
                                                        print(f"  Variable {var_idx_check} at position {pos_check} IS reachable, unfixing (was 0)")
                                                    reachable_vars.append((var_idx_check, coeff_check))
                                            
                                            # Apply fixes for non-reachable positions
                                            if vars_to_fix_zero:
                                                fix_dict = {v: 0 for v in vars_to_fix_zero}
                                                total_fixed.update(fix_dict)
                                                self.Q, _, log = self.reduce_qubo(fix_dict)
                                                self.reduction_log.extend(log)
                                            
                                            # Reverse reductions for positions that became reachable
                                            for var_to_unfix in vars_to_unfix:
                                                # Find the log entries for this variable
                                                self.Q, _ = self.reverse_reduction(self.Q, self.reduction_log, var_to_unfix)
                                                del total_fixed[var_to_unfix]
                                                print(f"  Unfixed variable {var_to_unfix}")
                                                # Add to reachable_vars if not already there
                                                if var_to_unfix not in [v[0] for v in reachable_vars]:
                                                    # Get the coefficient from Q
                                                    if (var_to_unfix, var_to_unfix) in self.Q:
                                                        reachable_vars.append((var_to_unfix, self.Q[(var_to_unfix, var_to_unfix)]))
                                            
                                            # If we have no reachable variables but we have reachable positions,
                                            # we need to search for variables that might have been fixed to 0 earlier
                                            if not reachable_vars and reachable_positions:
                                                print(f"  No reachable variables found in current QUBO, searching for previously fixed variables...")
                                                # Search through all possible variable indices for this robot at this timestep
                                                for pos in reachable_positions:
                                                    # Calculate the variable index for this position
                                                    # print("pos", pos)
                                                    if type == "grid":
                                                        i_pos, j_pos = pos
                                                        local_pos_idx = i_pos * N + j_pos
                                                    else:  # graph
                                                        # For graph, pos is the node ID directly
                                                        # local_pos_idx = self.problem.graph.get_node_from_position(pos)
                                                        local_pos_idx = pos
                                                    
                                                    var_idx_for_pos = robot_offset + t_check * vars_per_time + local_pos_idx
                                                    
                                                    # If this variable is not in robot_time_step_vars, it was reduced earlier
                                                    # Try to reverse the reduction to make it available
                                                    var_indices_in_timestep = [v[0] for v in robot_time_step_vars.get(t_check, [])]
                                                    if var_idx_for_pos not in var_indices_in_timestep:
                                                        print(f"  Variable {var_idx_for_pos} at position {pos} not in current QUBO, attempting to reverse reduction...")
                                                        # Try to reverse it from the instance log
                                                        self.Q, _ = self.reverse_reduction(self.Q, self.reduction_log, var_idx_for_pos)
                                                        
                                                        # Also remove from total_fixed if it's there
                                                        if var_idx_for_pos in total_fixed:
                                                            del total_fixed[var_idx_for_pos]
                                                            print(f"    Removed from total_fixed")
                                                        
                                                        # Check if it now exists in Q and add to reachable_vars
                                                        if (var_idx_for_pos, var_idx_for_pos) in self.Q:
                                                            reachable_vars.append((var_idx_for_pos, self.Q[(var_idx_for_pos, var_idx_for_pos)]))
                                                            print(f"    Successfully reversed, added to reachable_vars with coeff {self.Q[(var_idx_for_pos, var_idx_for_pos)]}")
                                                        else:
                                                            print(f"    Variable {var_idx_for_pos} still not in Q after reversal")
                                            
                                            # Now handle the reachable variables for this timestep
                                            # We need to fix one to 1 (the best one) and the rest to 0
                                            if reachable_vars:
                                                if len(reachable_vars) == 1:
                                                    # Only one reachable variable, fix it to 1
                                                    var_to_fix_one = reachable_vars[0][0]
                                                    total_fixed[var_to_fix_one] = 1
                                                    self.Q, _, log = self.reduce_qubo({var_to_fix_one: 1})
                                                    self.reduction_log.extend(log)
                                                    print(f"  Only one reachable variable {var_to_fix_one}, fixing to 1")
                                                    if t_check == t:
                                                        prev_fixed_pos = var_to_fix_one
                                                        prev_timestep = t_check
                                                else:
                                                    # Multiple reachable variables, use min_coeff logic
                                                    coeffs = [coeff for _, coeff in reachable_vars]
                                                    counts = Counter(coeffs)
                                                    if len(counts) > 1:
                                                        min_coeff_reach = min(counts)
                                                        if counts[min_coeff_reach] == 1:
                                                            # Fix the one with min coeff to 1, others to 0
                                                            fix_dict = {}
                                                            for var_reach, coeff_reach in reachable_vars:
                                                                if coeff_reach == min_coeff_reach:
                                                                    fix_dict[var_reach] = 1
                                                                    print(f"  Reachable variable {var_reach} has min coeff, fixing to 1")
                                                                    if t_check == t:
                                                                        prev_fixed_pos = var_reach
                                                                        prev_timestep = t_check
                                                                else:
                                                                    fix_dict[var_reach] = 0
                                                                    print(f"  Reachable variable {var_reach} has higher coeff, fixing to 0")
                                                            total_fixed.update(fix_dict)
                                                            self.Q, _, log = self.reduce_qubo(fix_dict)
                                                            self.reduction_log.extend(log)
                                            else:
                                                print(f"  WARNING: No reachable variables found for timestep {t_check}, this may cause missing timesteps!")
                                        
                                        bfs_adjusted = True
                                        break  # Break out of the for var_idx, coeff loop
                                    
                                if not bfs_adjusted:
                                    timestep_fixed[var_idx] = 1
                                    prev_fixed_pos = var_idx
                                    prev_timestep = t
                                    print(var_idx, f"fixed to 1 for robot {robot_id}, time step {t}")
                            else:
                                if not bfs_adjusted:
                                    timestep_fixed[var_idx] = 0
                    else:
                        # These are the worst (largest) coeffs, set to 0
                        max_coeff = max(counts)
                        # print("ddx")
                        if counts[max_coeff] >= 1:
                            for var_idx, coeff in robot_time_step_vars[t]:
                                if coeff == max_coeff:
                                    timestep_fixed[var_idx] = 0
                
                # Reduce QUBO after processing each timestep
                # This ensures adjacency constraints are updated before next timestep
                if timestep_fixed:
                    total_fixed.update(timestep_fixed)
                    self.Q, _, log = self.reduce_qubo(timestep_fixed)
                    self.reduction_log.extend(log)
                
                # If we did BFS adjustment, we've already handled all remaining timesteps
                # Break out of the outer loop to avoid KeyError on recalculated robot_time_step_vars
                if bfs_adjusted:
                    break

        return total_fixed

    def reduce_diag_fixed_vars_iterative(self, initial_reduction_log=None):
        """
        Iteratively apply diag_fixed_vars until no new fixed variables are found.
        
        Args:
            initial_reduction_log: Optional reduction log from previous reductions
                                  (e.g., from get_fixed_variables BFS fixes)

        Returns:
            dict: {variable_index: fixed_value} where fixed_value is 0 or 1
        """
        # Initialize reduction log for this reduction session
        # Include any initial reductions (e.g., from BFS in get_fixed_variables)
        self.reduction_log = initial_reduction_log if initial_reduction_log is not None else []
        
        total_fixed = {}
        while True:
            new_fixed = self.diag_fixed_vars()
            if not new_fixed:
                break
            total_fixed.update(new_fixed)
            self.Q, _, log = self.reduce_qubo(new_fixed)
            self.reduction_log.extend(log)
        
        # Clear reduction log after we're done to free memory
        self.reduction_log = []
        
        return total_fixed

    def list_to_dict_solution(self, solution_list):
        if isinstance(solution_list, list) and len(solution_list) > 0:
            solution_dict = (
                solution_list[0] if isinstance(solution_list[0], dict) else {}
            )
        elif isinstance(solution_list, dict):
            solution_dict = solution_list
        else:
            solution_dict = {}
        return solution_dict

    def reconstruct_solution(self, reduced_sol, fixed_vars, total_vars):
        if isinstance(reduced_sol, list):
            reduced_sol = self.list_to_dict_solution(reduced_sol)
        full_sol_dict = {}
        for i in range(total_vars):
            if i in fixed_vars:
                full_sol_dict[i] = np.int8(fixed_vars[i])
            elif i in reduced_sol:
                full_sol_dict[i] = np.int8(reduced_sol[i])
            else:
                full_sol_dict[i] = np.int8(0)
        return full_sol_dict

    def reachable_positions(self, robot, start_time, end_time):
        """
        Compute reachable positions per time step with backtracking.

        Args:
            robot: Robot object with current_position and robot_num.
            start_time (int): starting time step.
            end_time (int): ending time step (exclusive).

        Returns:
            dict[int, set[tuple[int, int]]]: {t: {(i, j), ...}} reachable positions per time step.
        """
        pass

    def reachable_positions_aggressive(self, robot, start, start_time, end_time):
        """
        Compute reachable positions per time step without backtracking.
        That means once a cell is reached, it won't be revisited in future time steps.

        Args:
            robot: Robot object with current_position and robot_num.
            start_time (int): starting time step.
            end_time (int): ending time step (exclusive).

        Returns:
            dict[int, set[tuple[int, int]]]: {t: {(i, j), ...}} reachable positions per time step.
        """
        pass
