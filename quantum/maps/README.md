# Map Files

This directory contains environment maps used for quantum path planning experiments.

## File Formats

Maps are stored in two formats:

1. **YAML (`.yaml`)** - Human-readable source format
   - Easy to edit and version control
   - Defines grid structure, obstacles, terrain, elevation
   - See `template.yaml` for format specification

2. **HDF5 (`.h5`)** - Binary runtime format
   - Efficient storage and loading
   - Generated automatically from YAML files
   - Includes both grid and graph representations

## Directory Structure

```
maps/
├── synthetic/          # Synthetic test maps
│   ├── 3x3/           # Small test cases
│   ├── 5x5/           # Medium complexity
│   ├── 10x10/         # Larger scenarios
│   └── ...
├── template.yaml      # Template for creating new maps
├── yaml2HDF5.py       # Conversion script
└── map_generator.py   # Programmatic map generation
```

## Generating Maps

### Generate All Maps

From the repository root:

```bash
python generate_all_maps.py
```

This will:
- Find all `.yaml` files in `quantum/maps/`
- Generate corresponding `.h5` files in the same directories
- Skip `template.yaml`

### Options

```bash
# Clean all .h5 files before regenerating
python generate_all_maps.py --clean

# Show detailed output
python generate_all_maps.py --verbose

# Both
python generate_all_maps.py --clean --verbose
```

### Generate Single Map

```python
from quantum.maps.yaml2HDF5 import generate_map_from_yaml

generate_map_from_yaml(
    yaml_path="quantum/maps/synthetic/3x3/my_map.yaml",
    output_dir="quantum/maps/synthetic/3x3"
)
```

## Version Control

Both YAML and HDF5 files are tracked in git:

- **YAML files** are the source of truth
- **HDF5 files** are committed for convenience (users don't need to generate them)
- `.gitignore` is configured to:
  - Exclude `.h5` files everywhere EXCEPT `quantum/maps/`
  - Allow map files to be committed while excluding generated data

**Why commit HDF5 files?**
- Users can clone and run immediately without setup
- Ensures consistency across different systems
- Small file sizes for test maps (<1 KB each)

**When to regenerate:**
- After modifying any `.yaml` file
- Before committing map changes
- When adding new maps

## Creating New Maps

1. **Copy the template:**
   ```bash
   cp quantum/maps/template.yaml quantum/maps/synthetic/my_map.yaml
   ```

2. **Edit the YAML file:**
   - Set map name, dimensions
   - Define obstacles, terrain, elevation
   - See template comments for guidance

3. **Generate HDF5:**
   ```bash
   python generate_all_maps.py
   ```

4. **Test the map:**
   ```python
   from quantum.pathFormulation import PathfindingProblem
   
   problem = PathfindingProblem.from_unified_data(
       "quantum/maps/synthetic/my_map.h5",
       start=(0, 0),
       end=(5, 5),
       T=10
   )
   ```

5. **Commit both files:**
   ```bash
   git add quantum/maps/synthetic/my_map.yaml
   git add quantum/maps/synthetic/my_map.h5
   git commit -m "Add new map: my_map"
   ```

## Map Categories

### Synthetic Maps

Programmatically generated test cases:

- **No obstacles** (`no_obs*.yaml`) - Open grids for baseline testing
- **With obstacles** (`obs*.yaml`) - Various obstacle configurations
- **Terrain** (`*_ter.yaml`) - Different material costs
- **Elevation** (`*_elev.yaml`) - Height-based costs
- **Mixed** (`*_mix.yaml`) - Combined features

### Size Categories

- **Tiny** (2x2, 3x2, 3x3) - Unit tests, quick validation
- **Small** (5x5) - Algorithm development
- **Medium** (10x10) - Standard benchmarks
- **Large** (100x100, 1000x1000) - Scalability testing

## Technical Details

### HDF5 Structure

Each `.h5` file contains:

```
map_file.h5
├── map_structure      # Occupancy grid (0=free, 1=obstacle)
├── terrain           # Material type IDs (optional)
├── elevation         # Height values (optional)
├── materials         # Material name list (optional)
└── graph/
    ├── nodes         # Node positions [(x,y), ...]
    └── edges         # Edges [(i,j,weight), ...]
```

### Metadata Attributes

- `map_name`: Identifier
- `grid_size`: "MxN" format
- `resolution`: Spatial resolution (default: 1.0)
- `generated_from`: Source YAML path
- `generated_at`: Timestamp

## Troubleshooting

### "Map file not found"
- Ensure you've run `generate_all_maps.py`
- Check the path is relative to repository root

### "Materials not found"
- Ensure `quantum/config/materials.yaml` exists
- Check material names match those defined in config

### "Invalid map structure"
- Validate YAML syntax
- Check grid dimensions match obstacle positions
- Ensure all positions are within bounds

## See Also

- [`template.yaml`](template.yaml) - Map format specification
- [`yaml2HDF5.py`](yaml2HDF5.py) - Conversion implementation
- [`../config/materials.yaml`](../config/materials.yaml) - Material definitions
