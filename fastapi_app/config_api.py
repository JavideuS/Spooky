# These are module-level mutable objects
# They start empty, and get populated during FastAPI lifespan
global_solver_configs = {}
global_penalties_params = {}


# Optional: expose a function to initialize them
def load_solver_configs(solvers: dict):
    global_solver_configs.clear()

    for backend, configs in solvers.items():
        for name, config in configs.items():
            key = f"{backend}.{name}"
            global_solver_configs[key] = config
