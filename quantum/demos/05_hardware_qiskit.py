"""
Demo 5: Qiskit Hardware Execution

Purpose: Run on real IBM Quantum hardware
Difficulty: Advanced
Prerequisites: 
  - IBM Quantum account (https://quantum.ibm.com/)
  - API token configured (see HARDWARE_SETUP.md)
  - Understanding of demo 01_quickstart.py

This demo shows how to execute quantum pathfinding on real quantum hardware
using IBM's Qiskit platform. Note: Hardware execution can take hours due to
queue times, so we use a very small problem.

IMPORTANT: This demo requires valid IBM Quantum credentials!
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pennylane import numpy as np
from quantum.pathFormulation import PathfindingProblem
from quantum.builder import GraphQUBO
from quantum.solvers import SolverFactory
import quantum.config.parser as config_parser
import os
import time
from qiskit_ibm_runtime import QiskitRuntimeService

def check_credentials():
    """Check if IBM Quantum credentials are configured."""
    try:
        # Try to initialize the service - this works with saved accounts OR environment variables
        QiskitRuntimeService()
        return True
    except Exception as e:
        print(f"\n⚠️  WARNING: Could not connect to IBM Quantum: {e}")
        print("\nTo run this demo, you need to configure your credentials:")
        print("  Option 1 (Recommended): Save account locally")
        print("     python -c \"from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum', token='MY_TOKEN')\"")
        print("\n  Option 2: Set environment variable")
        print("     export IBM_QUANTUM_TOKEN='your_token_here'")
        print("\nSee HARDWARE_SETUP.md for detailed instructions.")
        return False

def main():
    print("=" * 70)
    print("Demo 5: Qiskit Hardware Execution")
    print("=" * 70)
    
    # Check credentials
    print("\n[1/5] Checking IBM Quantum credentials...")
    if not check_credentials():
        print("\n❌ Cannot proceed without valid credentials.")
        print("   Run this demo after setting up your IBM Quantum account.")
        return
    
    print("    ✓ IBM Quantum credentials found")
    
    # Step 2: Create a VERY SMALL problem (hardware has limited qubits)
    print("\n[2/5] Creating small problem for hardware...")
    print("    Note: Real quantum hardware has limited qubits (~150)")
    print("    We use a tiny 3x3 grid to stay within limits.")
    
    # Define base path
    base_path = Path(__file__).parent.parent
    config_path = base_path / "config/config.yaml"
    
    penalty_sets = config_parser.load_config(str(config_path), sections=["penalty_sets"])
    penalties_conf = penalty_sets["penalty_sets"]["crash"]
    
    # Use the smallest possible problem
    map_path = base_path / "maps/synthetic/3x3/obs3x3_standard"
    problem = PathfindingProblem.from_map_config(
        str(map_path),
        problem_name="baseline"
    )
    
    print(f"    ✓ Map: {problem.grid.M}x{problem.grid.N}")
    print(f"    ✓ Timeline: {problem.T} timesteps")
    
    # Step 3: Build QUBO
    print("\n[3/5] Building QUBO...")
    p_graph = problem.as_graph_only()
    builder = GraphQUBO(p_graph, penalties=penalties_conf, name="hardware_demo", robot_window_limits={"Lucia": 6})
    print("    ✓ QUBO built")
    
    # Step 4: Configure Qiskit solver
    print("\n[4/5] Configuring Qiskit hardware solver...")
    
    # Initial parameters for QAOA
    init_params = np.array([[1.70579, 0.70321062], [0.49879231, 0.49412656]], requires_grad=True)
    
    # Create solver with Qiskit remote backend
    try:
        qiskit_solver = SolverFactory.create_solver(
            solver="pennylane",
            normalize_scale=2.0,
            num_reads="auto",
            layers=2,
            optimizer="QNG",
            opt_steps=30,
            device="qiskit.remote",  # This connects to IBM Quantum
            params=init_params
        )
        print("    ✓ Qiskit solver configured")
        print("    ✓ Device: qiskit.remote (IBM Quantum)")
    except Exception as e:
        print(f"    ❌ Failed to configure Qiskit solver: {e}")
        print("\n    This might be due to:")
        print("      - Invalid API token")
        print("      - Network connection issues")
        print("      - Qiskit not installed (pip install qiskit)")
        return
    
    # Step 5: Submit to hardware
    print("\n[5/5] Submitting to IBM Quantum hardware...")
    print("\n" + "⚠️ " * 20)
    print("WARNING: Hardware execution can take HOURS due to queue times!")
    print("Your job will be queued behind other users' jobs.")
    print("You can check job status at: https://quantum.ibm.com/jobs")
    print("⚠️ " * 20)
    
    response = input("\nDo you want to proceed? (yes/no): ")
    if response.lower() != "yes":
        print("\n✋ Hardware execution cancelled.")
        print("   You can run this demo anytime with valid credentials.")
        return
    
    print("\n📤 Submitting job to IBM Quantum...")
    start_time = time.time()
    
    try:
        solution = qiskit_solver.solve_qubo_smart(builder, False)
        
        elapsed_time = time.time() - start_time
        
        # Results
        print("\n" + "=" * 70)
        print("RESULTS FROM REAL QUANTUM HARDWARE")
        print("=" * 70)
        
        path = qiskit_solver.decode_path(solution["solution"], p_graph)
        energy = qiskit_solver.total_energy(solution)
        
        print(f"\n✓ Solution received!")
        print(f"  Total time (including queue): {elapsed_time/60:.1f} minutes")
        print(f"  Energy: {energy:.4f}")
        print(f"  Path length: {len(path)} steps")
        print(f"\n  Path: {path}")
        
        print("\n" + "=" * 70)
        print("✅ Hardware execution complete!")
        print("\nCongratulations! You just ran quantum pathfinding on real quantum hardware!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Hardware execution failed: {e}")
        print("\nCommon issues:")
        print("  - Queue timeout (try again later)")
        print("  - Library missing")
        print("  - Network issues")
        print("\nCheck IBM Quantum status: https://quantum.ibm.com/")

if __name__ == "__main__":
    main()
