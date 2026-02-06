from pennylane import numpy as np
from solvers import SolverFactory, DynamicSolver

from pathFormulation import PathfindingProblem
import config.parser as config_parser
from builder import QUBOBuilder, GraphQUBO
import benchmark.benchmark as benchmark
from config.hdf5parser import load_both_from_hdf5, load_map_from_hdf5, load_graph_from_hdf5
from robotConfiguration import RobotConfig
import pennylane as qml
from quantum.utils.logger import set_verbose_level, get_logger

# Load configuration
config = config_parser.load_config("config/config.yaml", sections=["penalty_sets", "verbose"])
penalties_conf = config["penalty_sets"]["crash"]

# Set verbose level from config (0=Silent, 1=Minimal, 2=Standard, 3=Debug)
verbose_level = config["verbose"]["level"]
set_verbose_level(1)

materials_data = config_parser.load_config("config/materials.yaml")["materials"]

# Fast initialization using map config
problem = PathfindingProblem.from_map_config(
    "maps/synthetic/10x10/obs10x10_hard",
    problem_name="four_robots",
    materials_data=materials_data
)
# problem.robots["Lucia"].priority = 3
# anja = RobotConfig("Anja", (4, 4), (0, 0), start_time=0, priority=1, safety_radius=0)
# problem.add_robot(anja)
# showmaker = RobotConfig("Showmaker", (0, 2), (2, 0), start_time=0, priority=1, safety_radius=0)
# problem.add_robot(showmaker)
# caps = RobotConfig("Caps", (0, 0), (2, 2), start_time=0, priority=1, safety_radius=0)
# problem.add_robot(caps)
# problem.add_robot(anja, True) # This alternative keeps predefined time when adding robots

# Choose QUBO builder based on problem format:
# For grid problems:
p_grid = problem.as_grid_only()
builder = QUBOBuilder(p_grid, penalties=penalties_conf, name="standard", distance_scaling="enhanced_linear", robot_window_limits={"robot_0": 5})

# For graph problems:
# p_graph = problem.as_graph_only()
# builder = GraphQUBO(p_graph, penalties=penalties_conf, name="graph_problem", robot_window_limits={"Lucia": 5})
# builder = GraphQUBO(p_graph, penalties=penalties_conf, name="graph_problem", robot_window_limits={"robot_0": 5})

# Solver (Quantum annealing)
dwave_solver = SolverFactory.create_solver(solver="dwave", normalize_scale=4, num_reads=4)
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
        device="lightning.gpu", params=init_params, verbose_level=verbose_level)

qiskit_hardware = SolverFactory.create_solver(
        solver="pennylane", normalize_scale=4.0, num_reads="auto",
        layers=2, optimizer="QNG", opt_steps=30,
        device="qiskit.remote", params=init_params, verbose_level=verbose_level)

# Q = builder.build()
# solution = dwave_solver.solve_qubo(builder, False)
# print("Solution:", solution["solution"])
# print("Path:", dwave_solver.decode_path(solution["solution"], problem))
# print(f"Energy: {dwave_solver.total_energy(solution):.4f}")

benchmark = benchmark.BenchmarkRunner(builder, dwave_solver, num_runs=10, level=2)
benchmark.run_build()
