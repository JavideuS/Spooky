import benchmark.benchmark
import solvers.DWave_solver as DWave_solver
import map
import pathFormulation
import config.parser as config_parser
import QUBOBuilder
import benchmark.benchmark as benchmark
from pennylane import numpy as np


# grid = map.Grid(M=3, N=3)
grid = map.Grid(M=3, N=3, obstacles=[(1, 1)])

conf = config_parser.load_config("config/config.yaml", sections=["problems", "penalty_sets", "solver"])
map_conf = conf["problems"]["grid_3x3_default"]
grid = map.Grid.from_dict(map_conf)

M = grid.M
N = grid.N
adjacency = grid.adjacency

# problem = pathFormulation.PathfindingProblem(grid, start=(2, 0), end=(0, 2))
problem = pathFormulation.PathfindingProblem.from_dict(map_conf)

s_i = problem.start[0]
s_j = problem.start[1]
e_i = problem.end[0]
e_j = problem.end[1]
T = problem.T

print("Adjacency Map:", grid.adjacency)

# Initialize QUBO dictionary
Q = {}

# Penalty weights
K_hot = 3     # Must have exactly one position per time step
K_adj = 1.5    # Must move to valid neighbor
K_start = 3    # Must start at given position
K_goal = 1.5   # Encourage reaching goal
K_lock = 1    # Discourage leaving goal once reached
penalties = {"K_hot": K_hot, "K_adj": K_adj, "K_start": K_start, "K_goal": K_goal, "K_lock": K_lock}
penalties_conf = conf["penalty_sets"]["standard2x"]


# Helper function to get variable index
def decode_position(idx):
    t = idx // (M * N)
    pos = idx % (M * N)
    i = pos // N
    j = pos % N
    return i, j, t


# QUBOBuilder = QUBOBuilder.QUBOBuilder(problem, penalties=penalties)
QUBOBuilder = QUBOBuilder.QUBOBuilder(problem, penalties=penalties_conf)


Q = QUBOBuilder.build()

# Solver (Quantum annealing)
solver = DWave_solver.QUBOSolver(normalize_scale=2.0, num_reads=10)

# solver_conf = conf["solver"]["dwave_qa"]
# solver = DWave_solver.QUBOSolver.from_config(solver_conf)

result = solver.solve_qubo(Q)
best_sample = result['solution']
min_energy = result['energy']

print("Best Solution:", best_sample)
print("Energy:", min_energy)

path = []
for idx in range(M * N * T):
    if best_sample.get(idx, 0) == 1:
        i, j, t = decode_position(idx)
        path.append((i, j, t))

print("Decoded Path:", sorted(path, key=lambda x: x[2]))

validation_result = benchmark.is_solution_valid(best_sample, problem)

if validation_result["valid"]:
    print(validation_result["message"])
else:
    print(validation_result["message"])
    print("Details:", validation_result["details"])

# Solver (QAOA)

# QUBO dictionary to matrix conversion
# # # Example usage of the function
# # print(f"Matrix shape: {Q_matrix.shape}")
# # print(f"Number of non-zero elements: {np.count_nonzero(Q_matrix)}")
# # print("First 10x10 submatrix:")
# # print(Q_matrix[:10, :10])

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