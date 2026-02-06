# Quantum Navigation Demos

Comprehensive examples demonstrating the quantum pathfinding system, from basic quickstart to advanced features and real quantum hardware execution.

## 🚀 Quick Start

**New to quantum pathfinding?** Start here:

```bash
cd quantum
python demos/01_quickstart.py
```

This will solve a simple pathfinding problem in ~30 seconds.

## 📚 Demo Overview

| # | Demo | Purpose | Difficulty | Hardware |
| # | Demo | Purpose | Difficulty | Time | Hardware |
|---|------|---------|------------|------|----------|
| 01 | [Quickstart](#01-quickstart) | Fast automatic initialization | ⭐ Beginner | 1 min | No |
| 02 | [Manual Setup](#02-manual-setup) | Step-by-step control | ⭐⭐ Intermediate | 5 min | No |
| 03 | [Visualization](#03-visualization) | Beautiful plots & animations | ⭐ Beginner | 2 min | No |
| 04 | [Multi-Robot](#04-multi-robot) | Collision avoidance | ⭐⭐ Intermediate | 10 min | No |
| 05 | [Hardware](#05-hardware-qiskit) | Real quantum hardware | ⭐⭐⭐ Advanced | 10 min | Yes (IBM) |
| 06 | [Advanced](#06-advanced-features) | Rapid experimentation | ⭐⭐⭐ Advanced | 15 min | No |

\* _Hardware execution time includes queue waiting (can be hours)_

## 📖 Detailed Descriptions

### 01: Quickstart

**File**: `01_quickstart.py`

The fastest way to get started. Uses automatic initialization with `from_map_config()` to solve a simple pathfinding problem.

**What you'll learn**:

- Loading problems with one line of code
- Basic solver usage (DWave)
- Interpreting results

**Run it**:

```bash
python demos/01_quickstart.py
```

**Expected output**: Solution path on a 5x5 grid in ~10 seconds

---

### 02: Manual Setup

**File**: `02_manual_setup.py`

Learn the internals by manually loading maps, configuring penalties, and initializing problems.

**What you'll learn**:

- Manual H5 file loading
- YAML configuration parsing
- Penalty set selection
- Grid vs Graph representations
- Fine-grained control

**Run it**:

```bash
python demos/02_manual_setup.py
```

**When to use**: When you need more control than automatic initialization provides

---

### 03: Visualization

**File**: `03_visualization.py`

Create beautiful, interactive visualizations of your pathfinding results.

**What you'll learn**:

- Static plots with current position
- Step-by-step animations
- Custom images for robots/obstacles
- Terrain rendering
- Exporting to HTML and PNG

**Run it**:

```bash
python demos/03_visualization.py
```

**Output files**:

- `output/static_path.html` - Interactive static plot
- `output/step_by_step_path.html` - Animated step-by-step

**Open in browser** to explore interactive features!

---

### 04: Multi-Robot

**File**: `04_multi_robot.py`

Coordinate multiple robots on the same map with collision avoidance.

**What you'll learn**:

- Adding multiple robots
- Setting priorities
- Configuring safety radii
- Staggered start times
- Robot-specific window limits

**Run it**:

```bash
python demos/04_multi_robot.py
```

**Scenario**: 3 robots navigating a 5x5 grid with different priorities and start times

---

### 05: Hardware (Qiskit)

**File**: `05_hardware_qiskit.py`

Execute on **real IBM Quantum hardware** (requires IBM Quantum account).

**Prerequisites**:

1. IBM Quantum account: https://quantum.ibm.com/
2. API token configured (see [HARDWARE_SETUP.md](HARDWARE_SETUP.md))

**What you'll learn**:

- Qiskit remote device configuration
- Queue time handling
- Hardware-appropriate problem sizing
- Error handling for hardware failures

**Run it**:

```bash
export IBM_QUANTUM_TOKEN='your_token_here'
python demos/05_hardware_qiskit.py
```

**⚠️ Warning**: Hardware execution can take **hours** due to queue times!

---

### 06: Advanced Features

**File**: `06_advanced_features.py`

Rapid experimentation with advanced features using the automatic/convenient API.

**What you'll learn**:

- Quick robot addition (one-liners)
- Easy solver switching
- Penalty set comparison
- Configuration experimentation
- Rapid prototyping workflow

**Run it**:

```bash
python demos/06_advanced_features.py
```

**Use case**: When you want to quickly test different configurations without manual boilerplate

**Note**: For manual control and detailed configuration, see demo 02

---

## 🎓 Learning Path

**Recommended order for learning**:

1. **Start**: `01_quickstart.py` - Get familiar with the basics
2. **Understand**: `02_manual_setup.py` - Learn what happens under the hood
3. **Visualize**: `03_visualization.py` - See your results
4. **Scale up**: `04_multi_robot.py` - Handle multiple agents
5. **Explore**: `06_advanced_features.py` - Advanced scenarios
6. **Deploy**: `05_hardware_qiskit.py` - Run on real quantum hardware (optional)

## 🛠️ Requirements

### Basic Demos (01-04, 06)

```bash
pip install pennylane numpy plotly pyyaml h5py
```

### Hardware (05)

```bash
pip install qiskit qiskit-ibm-runtime
```

## 📁 Output Files

Demos create output files in the `output/` directory:

```
quantum/output/
├── static_path.html           # From demo 03
├── static_path.png            # From demo 03
├── step_by_step_path.html     # From demo 03
└── step_by_step_path.png      # From demo 03
```

## 🐛 Troubleshooting

### "Module not found" errors

```bash
# Make sure you're in the quantum directory
cd quantum
python demos/01_quickstart.py
```

### "Map file not found"

```bash
# Generate map files from YAML
python maps/generate_all_maps.py
```

### Visualization PNG export fails

```bash
# Install kaleido for image export
pip install kaleido
```

### Hardware demo fails

- Check your IBM Quantum token is set correctly
- Verify network connection
- See [HARDWARE_SETUP.md](HARDWARE_SETUP.md) for detailed troubleshooting

## 📚 Additional Resources

- **Main README**: `../README.md` - Project overview
- **Map Documentation**: `../maps/README.md` - Creating custom maps
- **Hardware Setup**: `HARDWARE_SETUP.md` - IBM Quantum configuration
- **API Documentation**: Coming soon

## 💡 Tips

- **Start small**: Use small grids (3x3, 5x5) for testing
- **Visualize everything**: Always check your results with demo 03
- **Experiment**: Try different penalty sets and solver settings
- **Read the code**: Each demo is heavily commented for learning
- **Control output**: Adjust `verbose.level` in `config/config.yaml` to control console output:
  - `0` = Silent (errors only)
  - `1` = Minimal (essential info only)
  - `2` = Standard (default, recommended)
  - `3` = Debug (all details for troubleshooting)

## 🤝 Contributing

Found a bug or want to add a demo? Contributions welcome!

## 📄 License

See main project LICENSE file.
