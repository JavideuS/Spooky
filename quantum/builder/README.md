# QUBO Builders

This package provides the core logic for constructing Quadratic Unconstrained Binary Optimization (QUBO) models for quantum navigation problems. It translates pathfinding requirements into mathematical constraints compatible with quantum annealers and QAOA.

## Components

### 1. Base Builder (`BaseQUBO`)
The abstract base class that defines the common interface and shared utilities for all QUBO builders.

**Key Features:**
- **QUBO to Ising Conversion**: Converts QUBO dictionaries to Ising Hamiltonians for quantum backends.
- **Variable Reduction**: Optimizes problem size by identifying and fixing variables based on constraints.
- **Window Management**: Handles sliding window approaches for solving large problems in segments.
- **Solution Reconstruction**: Decodes raw quantum results back into path coordinates.

### 2. Grid vs Graph Builders

While both builders share the same underlying goal—finding optimal paths for robots—they approach the problem space differently.

#### Grid Builder (`GridQUBOBuilder`)
- **Approach**: Models the environment as a complete 2D grid where every cell is a potential node.
- **Use Case**: Best for dense, unstructured environments where robots can move freely in cardinal directions.
- **Features**:
  - Full spatial awareness (obstacles, terrain costs, elevation).
  - Adjacency is implicitly defined by the grid structure (up, down, left, right).

#### Graph Builder (`GraphQUBO`)
- **Approach**: Models the environment as a topology of connected nodes (a graph).
- **Use Case**: Ideal for sparse environments, road networks, or waypoint-based navigation.
- **Relationship to Grid**: A graph can be modeled as a grid (where nodes are grid cells), but it is not restricted to grid geometry.
- **Optimization**:
  - **Variable Count**: Generally starts with fewer variables than a full grid representation since it only considers defined nodes.
  - **Convergence**: With the system's pre-processing (variable reduction) steps, the effective problem size often becomes similar to the grid approach for equivalent maps.
- **Performance**: Uses different heuristics tailored to graph topology. While performance is generally comparable to the grid builder, it may behave slightly differently depending on the specific connectivity of the map.

## Usage

```python
from quantum.builder import GridQUBOBuilder, GraphQUBO

# Initialize builder with problem definition
# For Grid:
builder = GridQUBOBuilder(
    problem=my_problem_instance,
    penalties=my_penalties,
    window_max_steps=10
)

# For Graph:
builder = GraphQUBO(
    problem=my_problem_instance,
    penalties=my_penalties
)

# Build the QUBO model
qubo_dict = builder.build()
```

## Advanced Features

- **Iterative Reduction**: The builders employ smart heuristics to pre-solve parts of the problem (e.g., impossible moves) to reduce the qubit count required.
- **Dynamic Penalties**: Penalties can scale over time or distance to prevent local minima.
