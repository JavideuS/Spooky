# These are module-level mutable objects
# They start empty, and get populated during FastAPI lifespan
global_solver_configs = {}
global_penalties_params = {}


def _flatten_solvers(node: dict, prefix: str, out: dict) -> None:
    """
    Recursively flatten solvers.yaml into dotted keys. A dict is a leaf solver
    config once it has a "backend" key (injected by the _defaults anchors);
    anything else is treated as a grouping level (e.g. dwave.fast, or the
    extra pennylane.inference / pennylane.train grouping) and recursed into.
    """
    for name, value in node.items():
        path = f"{prefix}.{name}" if prefix else name
        if "backend" in value:
            out[path] = value
        else:
            _flatten_solvers(value, path, out)


# Optional: expose a function to initialize them
def load_solver_configs(solvers: dict):
    global_solver_configs.clear()
    _flatten_solvers(solvers, "", global_solver_configs)
