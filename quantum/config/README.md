# Configuration & Parsing

This module handles the loading, validation, and management of configurations for the Quantum Navigation system. It serves as the bridge between external definition files (YAML, HDF5) and the internal problem representations.

## Components

### 1. Parsers
- **`parser.py`**: The main entry point for parsing configuration files. It interprets the global settings and orchestrates specific parsers.
- **`hdf5parser.py`**: Specialized parser for HDF5 files, typically used for dense data like large maps, elevation grids, or pre-computed cost maps.

### 2. Configuration Files
The system relies on several key YAML configuration files:

- **`config.yaml`**: Global system settings, including:
  - Map paths and dimensions
  - Algorithm parameters
  - Windowing settings
  - **Verbose logging level** (controls console output verbosity)
- **`materials.yaml`**: Defines physical properties of the environment (e.g., terrain costs, traversability) used by the builders to calculate movement penalties.

### 3. Verbose Logging Configuration

The `verbose` section in `config.yaml` controls the amount of console output during solving:

```yaml
verbose:
  # 0 = Silent: No output except errors
  # 1 = Minimal: Only essential information (iterations, final results)
  # 2 = Standard: Standard progress information (default)
  # 3 = Debug: All debug information including optimization steps
  level: 2
```

**Verbosity Levels:**
- **0 (Silent)**: Only errors are printed
- **1 (Minimal)**: Essential information only (benchmark headers, warnings, final results)
- **2 (Standard)**: Standard progress (qubit counts, window adjustments, robot paths) - **Default**
- **3 (Debug)**: All debug information (optimization steps, BFS details, variable fixing)

## Usage

```python
from quantum.config import parser

# Load the complete system configuration
config = parser.load_config("config/config.yaml")

# Access specific sections
map_settings = config['map']
```

## Structure

The configuration module ensures that all parts of the system (Builders, Solvers, Visualizers) share a single source of truth for parameters, ensuring consistency across experiments.
