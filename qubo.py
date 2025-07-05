# Define constants
M = 3
N = 3
s_i = 2
s_j = 0
e_i = 0
e_j = 2

# import numpy as np
from pennylane import numpy as np


def manhattan_distance(i, j, k, l_coord):
    return abs(i - k) + abs(j - l_coord)


T = int(manhattan_distance(s_i, s_j, e_i, e_j) * 1.5)

def normalize_qubo(Q, scale=1.0):
    # Extract values
    values = np.array(list(Q.values()))

    # Compute min/max
    max_val = np.max(np.abs(values))
    if max_val == 0:
        return Q

    # Scale all values to [-scale, scale]
    scale_factor = scale / max_val
    return {k: v * scale_factor for k, v in Q.items()}

adjacency = {}

# Define valid movements: 4-connectivity (no diagonals)
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Precompute valid neighbors for each (i,j)
for i in range(M):
    for j in range(N):
        neighbors = []
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if 0 <= ni < M and 0 <= nj < N:
                # Skip if obstacle (e.g., at (1,1))
                # if (ni, nj) == (1, 1) or (ni, nj) == (1, 0):
                #     continue
                neighbors.append((ni, nj))
        adjacency[(i, j)] = neighbors

# print("Adjacency Map:", adjacency)

# Initialize QUBO dictionary
Q = {}

# Penalty weights
K_hot = 3     # Must have exactly one position per time step
K_adj = 1.5    # Must move to valid neighbor
K_start = 3    # Must start at given position
K_goal = 1.5   # Encourage reaching goal
K_lock = 1    # Discourage leaving goal once reached


# Helper function to get variable index
def decode_position(idx):
    t = idx // (M * N)
    pos = idx % (M * N)
    i = pos // N
    j = pos % N
    return i, j, t


def time_index(i, j, t):
    return i*N + j + M*N*t


Q = {}

# Constraint 1: One position per time step
for t in range(T):
    indices = [i*M + j + (M*N)*t for i in range(M) for j in range(N)]
    
    for n in indices:
        # Q[(n, n)] = Q.get((n, n), 0) + (-2 * K_hot)
        Q[(n, n)] = Q.get((n, n), 0) + (-1 * K_hot)
    
    for i, n in enumerate(indices):
        for m in indices[i+1:]:
            Q[(n, m)] = Q.get((n, m), 0) + 2 * K_hot


# Constraint 2: Movement must be to adjacent cells
# Be mind that all three approach need different constants to work well

# Reward for being in adjacent cells
for t in range(T-1):
    for i in range(M):
        for j in range(N):
            n = i*3 + j + 9*t
            Q[(n, n)] = Q.get((n, n), 0) + K_adj
            for (k, l_coord) in adjacency[(i, j)]:
                m = k*3 + l_coord + 9*(t+1)
                Q[(n, m)] = Q.get((n, m), 0) - K_adj

# # Penalize invalid transitions
# for t in range(T - 1):  # no movement after last time step
#     for i in range(M):
#         for j in range(N):
#             n = i*N + j + M*N*t  # use consistent indexing

#             # Look at all possible positions at next time step
#             for k in range(M):
#                 for l_coord in range(N):
#                     m = k*N + l_coord + M*N*(t+1)

#                     # Skip self-loop unless explicitly allowed
#                     if (k == i and l_coord == j):
#                         continue

#                     # Only penalize if (k,l) is NOT in adjacency[(i,j)]
#                     if (k, l_coord) not in adjacency[(i, j)]:
#                         Q[(n, m)] = Q.get((n, m), 0) + K_adj

# This case implements both reward and penalty systems (since having both individual loops together cause a bit conflict)
# i,kj terms gets added and substracted between the two individual loops
# So this is the mix of both approaches
# for t in range(T - 1):
#     for i in range(M):
#         for j in range(N):
#             n = i*N + j + M*N*t
#             for k in range(M):
#                 for l_coord in range(N):
#                     m = k*N + l_coord + M*N*(t+1)
#                     if (k, l_coord) in adjacency[(i, j)]:
#                         Q[(n, m)] = Q.get((n, m), 0) - K_adj  # Reward
#                     else:
#                         Q[(n, m)] = Q.get((n, m), 0) + K_adj  # Penalize

# Constraint 3: Start at given position
start_idx = time_index(s_i, s_j, 0)
Q[(start_idx, start_idx)] += -K_start

# Constraint 4: Follow the path to the goal
for t in range(1, T):
    goal_idx = time_index(e_i, e_j, t)
    Q[(goal_idx, goal_idx)] += -K_goal
# for t in range(1, T):
#     goal_idx = time_index(e_i, e_j, t)
#     time_factor = 1 + (T - t) / T  # stronger earlier
#     Q[(goal_idx, goal_idx)] -= K_goal * time_factor

# Constraint 5: Lock the goal position
for t in range(T - 1):  # up to T-2 to reference t+1
    g_t = time_index(e_i, e_j, t)
    g_t_next = time_index(e_i, e_j, t + 1)

    # Linear term: +K_lock * x_g_t
    Q[(g_t, g_t)] = Q.get((g_t, g_t), 0) + K_lock

    # Quadratic term: -K_lock * x_g_t * x_g_t_next
    Q[g_t, g_t_next] = Q.get((g_t, g_t_next), 0) - K_lock

Q = normalize_qubo(Q, scale=2)

# # Solver (Quantum annealing)
from dimod import BinaryQuadraticModel
from dimod import SimulatedAnnealingSampler

bqm = BinaryQuadraticModel.from_qubo(Q)
sampler = SimulatedAnnealingSampler()
response = sampler.sample(bqm, num_reads=12)

# Get best solution
best_solution = response.first.sample
best_energy = response.first.energy

print("Best Solution:", best_solution)
print("Energy:", best_energy)

path = []
for idx in range(M * N * T):
    if best_solution.get(idx, 0) == 1:
        i, j, t = decode_position(idx)
        path.append((i, j, t))

print("Decoded Path:", sorted(path, key=lambda x: x[2]))

# Solver (QAOA)

# def qubo_dict_to_matrix(qubo_dict, n_variables=None):
#     """
#     Convert a QUBO dictionary to a numpy matrix.
    
#     Args:
#         qubo_dict: Dictionary with (i, j) keys and float values
#         n_variables: Number of variables (if None, will be inferred from keys)
    
#     Returns:
#         numpy.ndarray: QUBO matrix of shape (n_variables, n_variables)
#     """
#     if n_variables is None:
#         # Find the maximum index from the dictionary keys
#         max_idx = 0
#         for (i, j) in qubo_dict.keys():
#             if isinstance(i, tuple):
#                 idx_i = i[0] if isinstance(i[0], int) else int(str(i[0])[1:])
#                 max_idx = max(max_idx, idx_i)
#             else:
#                 max_idx = max(max_idx, i)
#             if isinstance(j, tuple):
#                 idx_j = j[0] if isinstance(j[0], int) else int(str(j[0])[1:])
#                 max_idx = max(max_idx, idx_j)
#             else:
#                 max_idx = max(max_idx, j)
#         n_variables = max_idx + 1
    
#     # Create the matrix
#     matrix = np.zeros((n_variables, n_variables))
    
#     # Fill the matrix
#     for (i, j), val in qubo_dict.items():
#         # Handle both tuple keys and integer keys
#         if isinstance(i, tuple):
#             row_idx = i[0] if isinstance(i[0], int) else int(str(i[0])[1:])
#         else:
#             row_idx = i
            
#         if isinstance(j, tuple):
#             col_idx = j[0] if isinstance(j[0], int) else int(str(j[0])[1:])
#         else:
#             col_idx = j
        
#         # Ensure indices are within bounds
#         if 0 <= row_idx < n_variables and 0 <= col_idx < n_variables:
#             matrix[row_idx, col_idx] = val
    
#     return matrix


# # # Example usage of the function
# # print("\nUsing the conversion function:")
# Q_matrix = qubo_dict_to_matrix(Q)
# # Q_matrix = qubo_dict_to_matrix(Q, n_variables=M * N * T)
# # print(f"Matrix shape: {Q_matrix.shape}")
# # print(f"Number of non-zero elements: {np.count_nonzero(Q_matrix)}")
# # print("First 10x10 submatrix:")
# # print(Q_matrix[:10, :10])

# import pennylane as qml

# # Convert binary variables x_i ∈ {0,1} to spin variables σ_i ∈ {-1,1}
# # x_i = (1 + σ_i)/2

# def qubo_to_ising(Q):
#     n = Q.shape[0]
#     coeffs = []
#     observables = []

#     # Symmetrize Q
#     Q_sym = (Q + Q.T) / 2

#     # Process all unordered pairs (i <= j)
#     for i in range(n):
#         for j in range(i, n): # This helps avoiding double iteration over same pairs
#             coeff = Q_sym[i, j]
#             if i == j:
#                 # Linear term
#                 coeffs.append(coeff / 2)
#                 observables.append(qml.PauliZ(i))
#             else:
#                 # Quadratic term
#                 coeffs.append(coeff / 4)
#                 observables.append(qml.PauliZ(i) @ qml.PauliZ(j))

#     # Constant term
#     constant = np.sum(Q_sym) / 4 + np.trace(Q_sym) / 2

#     H = qml.Hamiltonian(coeffs, observables)
#     return H, constant


# H, offset = qubo_to_ising(Q_matrix)
# # print(H.simplify())

# # QAOA pennylane 

# # Step 1: Define the device
# dev = qml.device("default.qubit", wires=H.wires)

# # Step 2: Define the QAOA ansatz
# def qaoa_ansatz(params):
#     # Initial state: |+>
#     for w in dev.wires:
#         qml.Hadamard(wires=w)
    
#     # Time evolution under the Hamiltonian (QAOA layer)
#     qml.ApproxTimeEvolution(H, params[0], 1)

# # Step 3: Define the cost function using a QNode
# @qml.qnode(dev)
# def cost_function(params):
#     qaoa_ansatz(params)
#     return qml.expval(H)  # This includes all terms except the constant offset

# # Step 4: Optimize
# opt = qml.GradientDescentOptimizer(stepsize=0.5)
# params = np.array([0.1], requires_grad=True)
# steps = 100

# for i in range(steps):
#     params = opt.step(cost_function, params)
#     if i % 10 == 0:
#         print(f"Step {i}: Energy = {cost_function(params):.6f}")

# print("\nOptimized parameters:", params)
# print("Minimized energy (excluding offset):", cost_function(params))
# print("Total minimized energy (with offset):", cost_function(params) + offset)