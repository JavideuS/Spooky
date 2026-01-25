from pennylane import numpy as np
from solvers import SolverFactory, DynamicSolver

from pathFormulation import PathfindingProblem
import config.parser as config_parser
from builder import QUBOBuilder, GraphQUBO
import benchmark.benchmark as benchmark
from config.hdf5parser import load_both_from_hdf5, load_map_from_hdf5, load_graph_from_hdf5
import time
from robotConfiguration import RobotConfig
import pennylane as qml

conf = config_parser.load_config("config/config.yaml", sections=["problems", "penalty_sets"])

# map_hdf5 = load_both_from_hdf5("maps/synthetic/3x3/obs3x3_standard.h5")
# print("Double load from HDF5:", map_hdf5["map_data"])
# map_hdf5 = load_map_from_hdf5("maps/synthetic/3x3/obs3x3_standard.h5")
# print("Map loaded from HDF5:", map_hdf5)
# map_hdf5 = load_map_from_hdf5("maps/synthetic/3x2/no_obs3x2.h5")
# map_hdf5 = load_map_from_hdf5("maps/synthetic/5x5/obs5x5_hard.h5")
map_hdf5 = load_map_from_hdf5("maps/synthetic/10x10/no_obs10x10.h5")
# print("Map loaded from HDF5:", map_hdf5["name"])

# graph_data = load_graph_from_hdf5("maps/synthetic/3x3/obs3x3_standard.h5")
# graph = map.Graph.from_hdf5_data(graph_data)

# map_conf = conf["problems"]["grid_5x5_medium"]
# map_conf = conf["problems"]["grid_5x5_hard"]
# map_conf = conf["problems"]["grid_5x5_easyv2_alt"]
# map_conf = conf["problems"]["grid_3x3_default"]
map_conf = conf["problems"]["grid_10x10_no_obs"]
materials_data = config_parser.load_config("config/materials.yaml")["materials"]

# grid = map.Grid.from_hdf5_data(map_hdf5, materials_data)

# problem = pathFormulation.PathfindingProblem(grid, start=(2, 0), end=(0, 2))
# problem = pathFormulation.PathfindingProblem.from_dict(map_conf)
# problem = PathfindingProblem.from_grid_dict(grid, map_conf)
# problem = PathfindingProblem.from_graph_data(
#     graph_data, start_node=(2, 0), end_node=(0, 2), T=6
# )
# print("Problem:", map_conf)
start = map_conf["start"]
end = map_conf["goal"]
T = map_conf["T"]
problem = PathfindingProblem.from_unified_data(
        "maps/synthetic/10x10/no_obs10x10.h5",
        start=start,
        end=end,
        T=T,
        name="unified"
    )
problem.robots["Angie"].priority = 3
anja = RobotConfig("Anja", (4, 4), (0, 0), start_time=0, priority=1, safety_radius=0)
problem.add_robot(anja)
# showmaker = RobotConfig("Showmaker", (0, 2), (2, 0), start_time=0, priority=1, safety_radius=0)
# problem.add_robot(showmaker)
# caps = RobotConfig("Caps", (0, 0), (2, 2), start_time=0, priority=1, safety_radius=0)
# problem.add_robot(caps)
# problem.add_robot(anja, True) # This alternative keeps predefined time when adding robots
# print(anja.to_dict())
# print("Problem", problem.to_dict())


# print("Materials:", grid.materials)
# print("Problem ter:", problem.grid.terrain)
# print("Elevation Grid:", grid.elevation)

# print("Adjacency Map:", problem.graph.adjacency.get(0, []))
# print("Obstacles:", grid.obstacles)
material_costs = {
    "ceramic": 1.0,
    "water": 5.0,
    "asphalt": 1.5,
    "grass": 2.0
}
# problem.grid.material_cost = material_costs
# penalties_conf = conf["penalty_sets"]["graph"]
penalties_conf = conf["penalty_sets"]["crash"]

# Choose QUBO builder based on problem format:
# For grid problems:
# p_grid = problem.as_grid_only()
# builder = QUBOBuilder(p_grid, penalties=penalties_conf, name="standard", distance_scaling="enhanced_linear", robot_window_limits={"Angie": 4})

# For graph problems:
p_graph = problem.as_graph_only()
builder = GraphQUBO(p_graph, penalties=penalties_conf, name="graph_problem", robot_window_limits={"Angie": 3})

# start_time = time.time()
# Q = builder.build()
# duration = time.time() - start_time
# print(f"QUBO built in {duration:.4f} seconds")

# Solver (Quantum annealing)
dwave_solver = SolverFactory.create_solver(solver="dwave", normalize_scale=4, num_reads=7)
# Smart reads with pre-processing
# < 9 qubits -> 9 reads (Still not perfect)
# <= 18 qubits -> 12 reads
# Note that 9 reads also work fine for 18 qubits (12 errors on 1k attempts), but dwave tend to always fail even a bit
# 16 qubits -> 4.0 norm (also works for 32 qubits somehow??¿¿¿) (note that norm 3 works too but 4 seems to do better with less runs)

# init_params = np.array([[1.182179, 0.78571062], [0.36629231, 0.59672656]], requires_grad=True)
# init_params = np.array([[1.702179, 0.74571062], [0.45629231, 0.49672656]], requires_grad=True)
init_params = np.array([[1.70579, 0.70321062], [0.49879231, 0.49412656]], requires_grad=True)
pennylane_solver = SolverFactory.create_solver(
        solver="pennylane", normalize_scale=1.0, num_reads="auto",
        layers=2, optimizer="QNG", opt_steps=30,
        device="lightning.gpu", params=init_params)

qiskit_hardware = SolverFactory.create_solver(
        solver="pennylane", normalize_scale=4.0, num_reads="auto",
        layers=2, optimizer="QNG", opt_steps=30,
        device="qiskit.remote", params=init_params)

# Q = builder.build()
# solution = dwave_solver.solve_qubo(builder, False)
# print("Solution:", solution["solution"])
# print("Path:", dwave_solver.decode_path(solution["solution"], problem))
# print(f"Energy: {dwave_solver.total_energy(solution):.4f}")

benchmark = benchmark.BenchmarkRunner(builder, pennylane_solver, num_runs=1000, level=1)
benchmark.run_build()
