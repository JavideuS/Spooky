# To do:
# Automatize valid_positions and adjacency map generation

# Define constants
M = 3
N = 3
T = 6
s_i = 2
s_j = 0
e_i = 0
e_j = 2

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
                if (ni, nj) == (1, 1):
                    continue
                neighbors.append((ni, nj))
        adjacency[(i, j)] = neighbors


# Helper function to get variable index
def var_index(pos, t):
    return pos * num_time_steps + t  # Flatten pos-time into unique integer index

# Initialize QUBO dictionary
Q = {}



# Penalty weights
K_adj = 5.0
K_hot = 5.0
K_start = 5.0
K_goal = 3.0  # Reward for being at goal

# Helper function to get variable index
def var_index(pos, m,n):
    return pos % (M * N)

# Constraint 1: One position per time step
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



# # Constraint 2: Movement must be to adjacent cells
# for t in range(T):
#     for i in range(M):
#         for j in range(N):
#             n = i*3 + j + 9*t
#             Q[(n, n)] = Q.get((n, n), 0) + K_adj
#             for (k, l) in adjacency[(i, j)]:
#                 m = k*3 + l + 9*(t+1)
#                 Q[(n, m)] = Q.get((n, m), 0) - K_adj
# Assume valid_positions is a list of allowed (i,j) pairs (excluding obstacle)
# adjacency[(i,j)] contains list of allowed (k,l) positions

for t in range(T - 1):  # no movement after last time step
    for i in range(M):
        for j in range(N):
            n = i*N + j + M*N*t  # or whatever your indexing scheme is

            # Get all possible positions at next time step
            for k in range(M):
                for l in range(N):
                    if (k == i and l == j):
                        continue  # skip same cell; handled by self-loop or movement rules
                    m = k*N + l + M*N*(t+1)

                    # Only penalize if (k,l) is NOT in adjacency[(i,j)]
                    if (k, l) not in adjacency[(i, j)]:
                        Q[(n, m)] = Q.get((n, m), 0) + K_adj



# Constraint 3: Start at given position
start_idx = s_i*3 + s_j
end_idx = e_i*3 + e_j + 9*(T-1)
Q[start_idx, start_idx] += (-2 * K_start) + K_start
Q[end_idx, end_idx] += (-2 * K_start) + K_start


## Solver 
from dimod import BinaryQuadraticModel
from dimod import SimulatedAnnealingSampler

bqm = BinaryQuadraticModel.from_qubo(Q)
sampler = SimulatedAnnealingSampler()
response = sampler.sample(bqm, num_reads=10)

# Get best solution
best_solution = response.first.sample
best_energy = response.first.energy

print("Best Solution:", best_solution)
print("Energy:", best_energy)

path = best_solution.get(4, 0)
path = []
for idx in range(M * N * T):
    if best_solution.get(idx, 0) == 1:
        path.append(var_index(idx, M, N))

print("Path:", path)