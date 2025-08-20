import solvers.DWave_solver as DWave_solver
import map
import pathFormulation
import config.parser as config_parser
import QUBOBuilder
import benchmark.benchmark as benchmark
from config.hdf5parser import load_map_from_hdf5
import time

# grid = map.Grid(M=3, N=3)
# grid = map.Grid(M=3, N=3, obstacles=[(1, 1)])

conf = config_parser.load_config("config/config.yaml", sections=["problems", "penalty_sets"])

# map_hdf5 = load_map_from_hdf5("maps/synthetic/3x3/no_obs3x3_ter.h5")
# map_hdf5 = load_map_from_hdf5("maps/synthetic/3x3/no_obs3x3ter_alt.h5")
map_hdf5 = load_map_from_hdf5("maps/synthetic/3x3/no_obs3x3_elev.h5")
# print("Map loaded from HDF5:", map_hdf5["name"])

map_conf = conf["problems"]["grid_3x3_default"]
# map_conf = conf["problems"]["grid_5x5_easyv2_alt"]
# map_conf = conf["problems"]["grid_5x5_easyv2"]
grid = map.Grid.from_dict(map_conf)
grid = map.Grid.from_hdf5_data(map_hdf5)

# problem = pathFormulation.PathfindingProblem(grid, start=(2, 0), end=(0, 2))
# problem = pathFormulation.PathfindingProblem.from_dict(map_conf)
problem = pathFormulation.PathfindingProblem.from_grid_dict(grid, map_conf)
# print("Materials:", grid.materials)
# print("Problem ter:", problem.grid.terrain)
print("Elevation Grid:", grid.elevation)

print("Adjacency Map:", grid.adjacency)
print("Obstacles:", grid.obstacles)
material_costs = {
    "ceramic": 1.0,
    "water": 5.0,
    "asphalt": 1.5,
    "grass": 2.0
}
problem.grid.material_cost = material_costs
penalties_conf = conf["penalty_sets"]["elev"]
builder = QUBOBuilder.QUBOBuilder(problem, penalties=penalties_conf, name="standard")

# start_time = time.time()
# Q = QUBOBuilder.build()
# duration = time.time() - start_time
# print(f"QUBO built in {duration:.4f} seconds")

# Solver (Quantum annealing)
solver = DWave_solver.QUBOSolver(normalize_scale=2.0, num_reads=15)

# solver = DWave_solver.QUBOSolver.from_config(solver_conf)

# solution = solver.solve_qubo(QUBOBuilder)
# print("Solution:", solution["solution"])
# print("Path:", solver.decode_path(solution["solution"], problem))
# print(f"Energy: {solver.total_energy(solution):.4f}")

benchmark = benchmark.BenchmarkRunner(builder, solver, num_runs=100)
benchmark.run_build()