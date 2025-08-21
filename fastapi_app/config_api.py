# These are module-level mutable objects
# They start empty, and get populated during FastAPI lifespan
global_solver_configs = {}
global_penalties_params = {}
global_aliases = {}


# Optional: expose a function to initialize them
def load_solver_configs(solvers: dict, aliases: dict):
    global_solver_configs.clear()
    global_aliases.clear()

    # Registering solvers and aliases
    for backend, configs in solvers.items():
        for name, config in configs.items():
            key = f"{backend}.{name}"
            global_solver_configs[key] = config

    for alias, target in aliases.items():
        if target in global_solver_configs:
            global_aliases[alias] = target
        else:
            print(f"Warning: alias '{alias}' points to unknown '{target}'")
