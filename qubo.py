import solvers.DWave_solver as DWave_solver
import map
import pathFormulation
import config.parser as config_parser
import QUBOBuilder
import benchmark.benchmark as benchmark
from config.hdf5parser import load_map_from_hdf5
from pennylane import numpy as np
import time

# grid = map.Grid(M=3, N=3)
# grid = map.Grid(M=3, N=3, obstacles=[(1, 1)])

conf = config_parser.load_config("config/config.yaml", sections=["problems", "penalty_sets", "solver"])

# map_hdf5 = load_map_from_hdf5("maps/synthetic/3x3/no_obs3x3_ter.h5")
map_hdf5 = load_map_from_hdf5("maps/synthetic/3x3/no_obs3x3alt.h5")
# print("Map loaded from HDF5:", map_hdf5["name"])

map_conf = conf["problems"]["grid_3x3_default"]
# map_conf = conf["problems"]["grid_5x5_easyv2_alt"]
# map_conf = conf["problems"]["grid_5x5_easyv2"]
grid = map.Grid.from_dict(map_conf)
grid = map.Grid.from_hdf5_data(map_hdf5)

# print("Materials:", grid.materials)
# problem = pathFormulation.PathfindingProblem(grid, start=(2, 0), end=(0, 2))
# problem = pathFormulation.PathfindingProblem.from_dict(map_conf)
problem = pathFormulation.PathfindingProblem.from_hdf5_data(map_hdf5, map_conf)
print("Problem ter:", problem.grid.terrain)

print("Adjacency Map:", grid.adjacency)
print("Obstacles:", grid.obstacles)
material_costs = {
    "ceramic": 1.0,
    "water": 5.0,
    "asphalt": 1.5,
    "grass": 2.0
}
penalties_conf = conf["penalty_sets"]["ter"]
QUBOBuilder = QUBOBuilder.QUBOBuilder(problem, penalties=penalties_conf, name="standard", material_cost=material_costs)

# start_time = time.time()
# Q = QUBOBuilder.build()
# duration = time.time() - start_time
# print(f"QUBO built in {duration:.4f} seconds")

# Solver (Quantum annealing)
solver = DWave_solver.QUBOSolver(normalize_scale=2.0, num_reads=15)

# solver_conf = conf["solver"]["dwave_qa"]
# solver = DWave_solver.QUBOSolver.from_config(solver_conf)

# solution = solver.solve_qubo(QUBOBuilder)
# print("Solution:", solution["solution"])
# print("Path:", solver.decode_path(solution["solution"], problem))
# print(f"Energy: {solver.total_energy(solution):.4f}")

benchmark = benchmark.BenchmarkRunner(QUBOBuilder, solver, num_runs=100)
benchmark.run_build()

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