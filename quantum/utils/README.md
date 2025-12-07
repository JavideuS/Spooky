# Quantum Utilities

A collection of helper functions and common tools used across the Quantum Navigation stack.

## Modules

### `paths.py`
Utilities for file system path management and directory resolution. Ensures that resources like maps, configs, and logs are correctly located regardless of the execution context.

## Usage

These utilities are primarily for internal use by the `builder` and `solver` modules but can be imported for custom extensions.

```python
from quantum.utils import paths

# Get the absolute path to the maps directory
maps_dir = paths.get_maps_dir()
```
