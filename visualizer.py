import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import time

class QuantumRoboticsVisualizer:
    """
    Simple visualization tools for quantum robotics framework
    No GUI required - generates static plots and animations
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('dark_background')  # Modern dark theme
        
    def visualize_grid_and_path(self, obstacles: List[Tuple[int, int]], 
                               width: int, height: int,
                               path: List[Tuple[int, int, int]], 
                               start: Tuple[int, int], goal: Tuple[int, int],
                               title: str = "Quantum Pathfinding Result",
                               save_path: Optional[str] = None):
        """
        Visualize grid map with obstacles and the computed path
        
        Args:
            obstacles: List of (x, y) coordinates representing obstacles
            width: Grid width
            height: Grid height
            path: List of (x, y, t) coordinates representing the path with time steps
            start: Starting position (x, y)
            goal: Goal position (x, y)
            title: Plot title
            save_path: If provided, saves the plot to this path
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create grid from obstacle list
        display_grid = np.zeros((height, width))
        
        # Mark obstacles
        for x, y in obstacles:
            if 0 <= x < width and 0 <= y < height:
                display_grid[y, x] = 1.0  # Obstacle cells
        
        # Extract spatial coordinates from path (ignore time)
        spatial_path = [(x, y) for x, y, t in path]
        
        # Mark path
        for x, y in spatial_path:
            if (x, y) != start and (x, y) != goal and 0 <= x < width and 0 <= y < height:
                display_grid[y, x] = 0.5  # Path cells
        
        # Mark start and goal
        if 0 <= start[0] < width and 0 <= start[1] < height:
            display_grid[start[1], start[0]] = 0.3  # Start
        if 0 <= goal[0] < width and 0 <= goal[1] < height:
            display_grid[goal[1], goal[0]] = 0.7   # Goal
        
        # Plot grid with custom colors
        im = ax.imshow(display_grid, cmap='viridis', alpha=0.8)
        
        # Overlay path as arrows with time annotations
        if len(path) > 1:
            for i in range(len(path) - 1):
                x1, y1, t1 = path[i]
                x2, y2, t2 = path[i + 1]
                
                # Draw arrow
                ax.arrow(x1, y1, x2-x1, y2-y1, 
                        head_width=0.1, head_length=0.1, 
                        fc='yellow', ec='yellow', alpha=0.8)
                
                # Add time step annotation
                ax.text(x1, y1-0.3, f't{t1}', fontsize=8, ha='center', 
                       color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            # Add final time step
            if path:
                x_final, y_final, t_final = path[-1]
                ax.text(x_final, y_final-0.3, f't{t_final}', fontsize=8, ha='center', 
                       color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Mark start and goal with special markers
        ax.plot(start[0], start[1], 'go', markersize=15, label='Start')
        ax.plot(goal[0], goal[1], 'ro', markersize=15, label='Goal')
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Labels and title
        ax.set_title(title, fontsize=16, fontweight='bold', color='white')
        ax.set_xlabel('X Coordinate', fontweight='bold')
        ax.set_ylabel('Y Coordinate', fontweight='bold')
        ax.legend()
        
        # Add path info with timing
        path_length = len(path)
        total_time = path[-1][2] + 1 if path else 0  # Time steps (0-indexed)
        ax.text(0.02, 0.98, f'Path: {path_length} steps\nTime: {total_time} units', 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        
        plt.show()
        
    def compare_algorithms(self, results: Dict[str, Dict], 
                          save_path: Optional[str] = None):
        """
        Compare performance between quantum and classical algorithms
        
        Args:
            results: Dict with algorithm names as keys and results as values
                   Each result should have: 'time', 'path_length', 'path'
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Quantum vs Classical Algorithm Comparison', fontsize=16, fontweight='bold')
        
        algorithms = list(results.keys())
        times = [results[alg]['time'] for alg in algorithms]
        path_lengths = [len(results[alg]['path']) for alg in algorithms]
        
        # Execution time comparison
        bars1 = ax1.bar(algorithms, times, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax1.set_title('Execution Time', fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        for bar, time_val in zip(bars1, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # Path length comparison  
        path_lengths = [len(results[alg]['path']) for alg in algorithms]
        path_times = [results[alg]['path'][-1][2] + 1 if results[alg]['path'] else 0 for alg in algorithms]  # Total time steps
        
        bars2 = ax2.bar(algorithms, path_lengths, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax2.set_title('Path Length (Steps)', fontweight='bold')
        ax2.set_ylabel('Number of Steps')
        for bar, length in zip(bars2, path_lengths):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{length}', ha='center', va='bottom')
        
        # Efficiency metric (steps per second)
        efficiency = [path_lengths[i] / max(times[i], 0.001) for i in range(len(algorithms))]
        bars3 = ax3.bar(algorithms, efficiency, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax3.set_title('Efficiency (Steps/Second)', fontweight='bold')
        ax3.set_ylabel('Steps per Second')
        for bar, eff in zip(bars3, efficiency):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{eff:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        
        plt.show()
        
    def animate_pathfinding(self, obstacles: List[Tuple[int, int]], 
                           width: int, height: int,
                           path: List[Tuple[int, int, int]], 
                           start: Tuple[int, int], goal: Tuple[int, int],
                           title: str = "Quantum Pathfinding Animation",
                           save_path: Optional[str] = None,
                           jupyter_mode: bool = False):
        """
        Create an animated visualization of the pathfinding process
        Shows the robot moving through time steps
        
        Args:
            jupyter_mode: Set to True when running in Jupyter notebooks for proper display
        """
        # Configure for Jupyter if needed
        if jupyter_mode:
            try:
                from IPython.display import HTML
                plt.rcParams['animation.html'] = 'jshtml'  # Use JavaScript HTML for animations
            except ImportError:
                print("Warning: IPython not available. Animation may not display properly in Jupyter.")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Setup the grid display from obstacles
        display_grid = np.zeros((height, width))
        
        # Mark obstacles
        for x, y in obstacles:
            if 0 <= x < width and 0 <= y < height:
                display_grid[y, x] = 1.0
        
        # Mark start and goal
        if 0 <= start[0] < width and 0 <= start[1] < height:
            display_grid[start[1], start[0]] = 0.3  # Start
        if 0 <= goal[0] < width and 0 <= goal[1] < height:
            display_grid[goal[1], goal[0]] = 0.7   # Goal
        
        im = ax.imshow(display_grid, cmap='viridis', alpha=0.8)
        
        # Initialize empty path line and current position marker
        line, = ax.plot([], [], 'yo-', linewidth=3, markersize=6, alpha=0.8, label='Path')
        current_pos, = ax.plot([], [], 'ro', markersize=12, alpha=0.9, label='Current Position')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontweight='bold')
        ax.set_ylabel('Y Coordinate', fontweight='bold')
        ax.legend()
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        
        def animate(frame):
            if frame < len(path):
                # Show path up to current frame
                current_path = path[:frame+1]
                x_coords = [p[0] for p in current_path]
                y_coords = [p[1] for p in current_path]
                line.set_data(x_coords, y_coords)
                
                # Show current position
                current_x, current_y, current_t = path[frame]
                current_pos.set_data([current_x], [current_y])
                
                # Update title with progress and time step
                ax.set_title(f'{title} - Step {frame+1}/{len(path)} (t={current_t})', 
                           fontsize=16, fontweight='bold')
            return line, current_pos
        
        anim = FuncAnimation(fig, animate, frames=len(path), 
                           interval=800, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path.replace('.png', '.gif'), writer='pillow', fps=1.2)
        
        if jupyter_mode:
            plt.close(fig)  # Prevent static display
            return anim  # Return animation object for Jupyter display
        else:
            plt.show()
            return anim
    
    def visualize_performance_scaling(self, grid_sizes: List[int], 
                                    quantum_times: List[float], 
                                    classical_times: List[float],
                                    save_path: Optional[str] = None):
        """
        Visualize how performance scales with problem size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear scale
        ax1.plot(grid_sizes, quantum_times, 'o-', label='Quantum (QAOA)', 
                linewidth=3, markersize=8, color='#ff6b6b')
        ax1.plot(grid_sizes, classical_times, 's-', label='Classical (A*)', 
                linewidth=3, markersize=8, color='#4ecdc4')
        ax1.set_xlabel('Grid Size (NxN)', fontweight='bold')
        ax1.set_ylabel('Execution Time (seconds)', fontweight='bold')
        ax1.set_title('Performance Scaling - Linear', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Log scale
        ax2.loglog(grid_sizes, quantum_times, 'o-', label='Quantum (QAOA)', 
                  linewidth=3, markersize=8, color='#ff6b6b')
        ax2.loglog(grid_sizes, classical_times, 's-', label='Classical (A*)', 
                  linewidth=3, markersize=8, color='#4ecdc4')
        ax2.set_xlabel('Grid Size (NxN)', fontweight='bold')
        ax2.set_ylabel('Execution Time (seconds)', fontweight='bold')
        ax2.set_title('Performance Scaling - Log Scale', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        
        plt.show()

# Usage example and helper functions
def create_sample_obstacles(width: int = 8, height: int = 8, 
                          obstacle_prob: float = 0.2) -> List[Tuple[int, int]]:
    """Create a sample obstacle list"""
    np.random.seed(42)  # For reproducible results
    obstacles = []
    
    for x in range(width):
        for y in range(height):
            # Skip start and goal positions
            if (x, y) == (0, 0) or (x, y) == (width-1, height-1):
                continue
            if np.random.random() < obstacle_prob:
                obstacles.append((x, y))
    
    return obstacles

def demo_visualization():
    """Demonstrate the visualization capabilities"""
    # Create visualizer
    viz = QuantumRoboticsVisualizer()
    
    # Create sample data - using obstacle list format
    width, height = 8, 8
    obstacles = create_sample_obstacles(width, height, 0.25)
    
    # Path with time steps (x, y, t) - like your QUBO output format
    path = [(0, 0, 0), (0, 1, 1), (1, 1, 2), (1, 2, 3), (2, 2, 4), 
            (3, 2, 5), (4, 2, 6), (5, 2, 7), (6, 2, 8), (7, 2, 9), 
            (7, 3, 10), (7, 4, 11), (7, 5, 12), (7, 6, 13), (7, 7, 14)]
    
    start = (0, 0)
    goal = (7, 7)
    
    # Visualize path
    viz.visualize_grid_and_path(obstacles, width, height, path, start, goal, 
                               title="Quantum QAOA Pathfinding Result")
    
    # Compare algorithms - note: paths need to be in (x,y,t) format
    classical_path = [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4), 
                     (5, 0, 5), (6, 0, 6), (7, 0, 7), (7, 1, 8), (7, 2, 9), 
                     (7, 3, 10), (7, 4, 11), (7, 5, 12), (7, 6, 13), (7, 7, 14)]
    
    results = {
        'QAOA': {'time': 3.142, 'path': path},
        'A*': {'time': 0.001, 'path': classical_path},
        'Dijkstra': {'time': 0.003, 'path': path}
    }
    viz.compare_algorithms(results)
    
    # Performance scaling
    grid_sizes = [3, 4, 5, 6, 7, 8]
    quantum_times = [0.1, 0.5, 3.0, 15.0, 45.0, 120.0]  # Exponential growth
    classical_times = [0.001, 0.002, 0.003, 0.005, 0.008, 0.012]  # Linear growth
    
    viz.visualize_performance_scaling(grid_sizes, quantum_times, classical_times)

if __name__ == "__main__":
    # demo_visualization()
    # Your QUBO output format
    path = [(2, 0, 0), (2, 1, 1), (2, 2, 2), (1, 2, 3), (0, 2, 4), (0, 2, 5)]
    obstacles = [(1, 1), (3, 2)]
    width, height = 4, 3
    start = (2, 0)
    goal = (0, 2)

    # Visualize
    viz = QuantumRoboticsVisualizer()
    viz.animate_pathfinding(obstacles, width, height, path, start, goal,
                            title="QUBO Solution with Time Steps")