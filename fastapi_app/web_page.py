import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

class QuantumRoboticsDashboard:
    """
    Interactive dashboard for quantum robotics visualization
    Combines your Plotly visualizer with interactive controls
    """
    
    def __init__(self, port=8050):
        self.app = dash.Dash(__name__)
        self.port = port
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup the dashboard layout with controls and visualizations"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🤖 Quantum Robotics Pathfinding Dashboard", 
                       className="dashboard-title"),
                html.P("Interactive visualization of quantum pathfinding algorithms",
                       className="dashboard-subtitle")
            ], className="header"),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Grid Size:"),
                    dcc.Slider(
                        id='grid-size-slider',
                        min=5, max=20, step=1, value=10,
                        marks={i: str(i) for i in range(5, 21, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Obstacle Density:"),
                    dcc.Slider(
                        id='obstacle-density-slider',
                        min=0.1, max=0.4, step=0.05, value=0.2,
                        marks={i/10: f"{i}%" for i in range(1, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Algorithm:"),
                    dcc.Dropdown(
                        id='algorithm-dropdown',
                        options=[
                            {'label': '🔮 Quantum QAOA', 'value': 'qaoa'},
                            {'label': '⚡ A* (Classical)', 'value': 'astar'},
                            {'label': '🌊 Dijkstra', 'value': 'dijkstra'},
                            {'label': '📊 Compare All', 'value': 'compare'}
                        ],
                        value='qaoa',
                        className="dropdown"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Visualization Mode:"),
                    dcc.RadioItems(
                        id='viz-mode-radio',
                        options=[
                            {'label': '🎬 Animated', 'value': 'animated'},
                            {'label': '📸 Static', 'value': 'static'},
                            {'label': '👣 Step-by-step', 'value': 'steps'},
                            {'label': '🏔️ 3D Elevation', 'value': '3d'}
                        ],
                        value='animated',
                        className="radio-items"
                    )
                ], className="control-item"),
                
                html.Button("🚀 Generate New Path", id="generate-button", 
                           className="generate-btn")
                
            ], className="control-panel"),
            
            # Main Visualization Area
            html.Div([
                dcc.Graph(
                    id='main-visualization',
                    style={'height': '600px'},
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'quantum_pathfinding',
                            'height': 600,
                            'width': 800,
                            'scale': 2
                        }
                    }
                )
            ], className="main-viz"),
            
            # Performance Metrics
            html.Div([
                html.Div(id="performance-metrics", className="metrics-container")
            ], className="metrics-panel"),
            
            # Algorithm Information
            html.Div([
                html.H3("Algorithm Information"),
                html.Div(id="algorithm-info", className="info-panel")
            ], className="info-section"),
            
            # Hidden div to store data
            html.Div(id='stored-data', style={'display': 'none'})
            
        ], className="dashboard-container")
        
        # Add custom CSS
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
        <head>
            {%metas%}
            <title>Quantum Robotics Dashboard</title>
            {%favicon%}
            {%css%}
            <style>
                .dashboard-container {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }
                .header {
                    text-align: center;
                    color: white;
                    margin-bottom: 30px;
                }
                .dashboard-title {
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }
                .dashboard-subtitle {
                    font-size: 1.2em;
                    opacity: 0.9;
                }
                .control-panel {
                    background: rgba(255,255,255,0.1);
                    padding: 20px;
                    border-radius: 15px;
                    margin-bottom: 20px;
                    backdrop-filter: blur(10px);
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    align-items: center;
                }
                .control-item {
                    flex: 1;
                    min-width: 200px;
                    color: white;
                }
                .generate-btn {
                    background: #ff6b6b;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 25px;
                    font-size: 16px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                .generate-btn:hover {
                    background: #ff5252;
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }
                .main-viz {
                    background: rgba(255,255,255,0.95);
                    padding: 20px;
                    border-radius: 15px;
                    margin-bottom: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                }
                .metrics-panel, .info-section {
                    background: rgba(255,255,255,0.9);
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                    backdrop-filter: blur(5px);
                }
                .metrics-container {
                    display: flex;
                    justify-content: space-around;
                    flex-wrap: wrap;
                    gap: 20px;
                }
                .metric-card {
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    min-width: 120px;
                }
                .info-panel {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #007bff;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
        </html>
        '''
    
    def setup_callbacks(self):
        """Setup interactive callbacks"""
        
        @self.app.callback(
            [Output('main-visualization', 'figure'),
             Output('performance-metrics', 'children'),
             Output('algorithm-info', 'children')],
            [Input('generate-button', 'n_clicks'),
             Input('grid-size-slider', 'value'),
             Input('obstacle-density-slider', 'value'),
             Input('algorithm-dropdown', 'value'),
             Input('viz-mode-radio', 'value')]
        )
        def update_visualization(n_clicks, grid_size, obstacle_density, algorithm, viz_mode):
            # Generate sample data (replace with your actual algorithms)
            obstacles = self.generate_obstacles(grid_size, grid_size, obstacle_density)
            path, metrics = self.simulate_algorithm(grid_size, grid_size, obstacles, algorithm)
            
            # Create visualization based on mode
            if viz_mode == '3d':
                fig = self.create_3d_visualization(grid_size, grid_size, obstacles, path)
            elif viz_mode == 'steps':
                fig = self.create_step_visualization(grid_size, grid_size, obstacles, path)
            elif viz_mode == 'animated':
                fig = self.create_animated_visualization(grid_size, grid_size, obstacles, path)
            else:  # static
                fig = self.create_static_visualization(grid_size, grid_size, obstacles, path)
            
            # Performance metrics
            metrics_cards = self.create_metrics_cards(metrics)
            
            # Algorithm info
            algo_info = self.get_algorithm_info(algorithm)
            
            return fig, metrics_cards, algo_info
    
    def generate_obstacles(self, width: int, height: int, density: float) -> List[Tuple[int, int]]:
        """Generate random obstacles"""
        np.random.seed(42)  # Reproducible for demo
        obstacles = []
        for x in range(width):
            for y in range(height):
                if (x, y) not in [(0, 0), (width-1, height-1)] and np.random.random() < density:
                    obstacles.append((x, y))
        return obstacles
    
    def simulate_algorithm(self, width: int, height: int, obstacles: List[Tuple[int, int]], 
                          algorithm: str) -> Tuple[List[Tuple[int, int, int]], Dict]:
        """Simulate pathfinding algorithm (replace with your actual implementation)"""
        # Simple diagonal path for demo
        path = []
        t = 0
        for i in range(min(width, height)):
            if (i, i) not in obstacles:
                path.append((i, i, t))
                t += 1
        
        # Add remaining steps to reach goal
        if path:
            last_x, last_y, _ = path[-1]
            while last_x < width - 1:
                last_x += 1
                path.append((last_x, last_y, t))
                t += 1
            while last_y < height - 1:
                last_y += 1
                path.append((last_x, last_y, t))
                t += 1
        
        # Simulate metrics
        metrics = {
            'execution_time': np.random.uniform(0.1, 5.0),
            'path_length': len(path),
            'nodes_explored': width * height // 2,
            'optimality': np.random.uniform(0.8, 1.0),
            'quantum_advantage': algorithm == 'qaoa'
        }
        
        return path, metrics
    
    def create_static_visualization(self, width: int, height: int, 
                                   obstacles: List[Tuple[int, int]], 
                                   path: List[Tuple[int, int, int]]) -> go.Figure:
        """Create static visualization"""
        fig = go.Figure()
        
        # Grid background
        grid = np.zeros((height, width))
        for x, y in obstacles:
            if 0 <= x < width and 0 <= y < height:
                grid[y, x] = 1.0
        
        fig.add_trace(go.Heatmap(
            z=grid,
            colorscale=[[0, 'rgba(240,248,255,0.8)'], [1, 'rgba(25,25,112,0.8)']],
            showscale=False,
            name='Grid'
        ))
        
        # Path
        if path:
            path_x, path_y = zip(*[(p[0], p[1]) for p in path])
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                mode='lines+markers',
                line=dict(color='#ff6b6b', width=4),
                marker=dict(size=8, color='#ff9999'),
                name='Quantum Path'
            ))
            
            # Start and goal
            fig.add_trace(go.Scatter(
                x=[path[0][0]], y=[path[0][1]],
                mode='markers+text',
                marker=dict(size=20, color='green', symbol='star'),
                text=['🚀'], textfont=dict(size=16),
                name='Start'
            ))
            
            fig.add_trace(go.Scatter(
                x=[path[-1][0]], y=[path[-1][1]],
                mode='markers+text',
                marker=dict(size=20, color='red', symbol='diamond'),
                text=['🎯'], textfont=dict(size=16),
                name='Goal'
            ))
        
        fig.update_layout(
            title="Quantum Pathfinding Visualization",
            xaxis=dict(title="X Coordinate", range=[-0.5, width-0.5]),
            yaxis=dict(title="Y Coordinate", range=[-0.5, height-0.5]),
            template='plotly_white'
        )
        
        return fig
    
    def create_3d_visualization(self, width: int, height: int,
                               obstacles: List[Tuple[int, int]],
                               path: List[Tuple[int, int, int]]) -> go.Figure:
        """Create 3D elevation visualization"""
        # Generate elevation data
        X, Y = np.meshgrid(range(width), range(height))
        elevation = 5 * np.sin(X/3) * np.cos(Y/3) + 2 * np.random.random((height, width))
        
        fig = go.Figure()
        
        # 3D surface
        fig.add_trace(go.Surface(
            z=elevation,
            x=X, y=Y,
            colorscale='terrain',
            opacity=0.8,
            name='Terrain'
        ))
        
        # 3D path
        if path:
            path_x, path_y = zip(*[(p[0], p[1]) for p in path])
            path_z = [elevation[min(p[1], height-1), min(p[0], width-1)] + 0.5 for p in path]
            
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines+markers',
                line=dict(color='red', width=6),
                marker=dict(size=5, color='red'),
                name='Path'
            ))
        
        fig.update_layout(
            title="3D Terrain Pathfinding",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Elevation"
            ),
            template='plotly_dark'
        )
        
        return fig
    
    def create_step_visualization(self, width: int, height: int,
                                 obstacles: List[Tuple[int, int]],
                                 path: List[Tuple[int, int, int]]) -> go.Figure:
        """Create step-by-step visualization"""
        if not path:
            return go.Figure()
        
        n_steps = min(6, len(path))
        step_indices = np.linspace(0, len(path)-1, n_steps, dtype=int)
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"Step {i+1}" for i in step_indices[:6]],
            specs=[[{"type": "scatter"} for _ in range(3)] for _ in range(2)]
        )
        
        for idx, step_i in enumerate(step_indices):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            # Current path
            current_path = path[:step_i+1]
            if current_path:
                path_x, path_y = zip(*[(p[0], p[1]) for p in current_path])
                
                fig.add_trace(go.Scatter(
                    x=path_x, y=path_y,
                    mode='lines+markers',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ), row=row, col=col)
        
        fig.update_layout(
            title="Algorithm Progression",
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_animated_visualization(self, width: int, height: int,
                                     obstacles: List[Tuple[int, int]],
                                     path: List[Tuple[int, int, int]]) -> go.Figure:
        """Create animated visualization"""
        # This would use your existing Plotly animation code
        return self.create_static_visualization(width, height, obstacles, path)
    
    def create_metrics_cards(self, metrics: Dict) -> List[html.Div]:
        """Create performance metrics cards"""
        cards = []
        
        cards.append(html.Div([
            html.H4(f"{metrics['execution_time']:.3f}s"),
            html.P("Execution Time")
        ], className="metric-card"))
        
        cards.append(html.Div([
            html.H4(f"{metrics['path_length']}"),
            html.P("Path Length")
        ], className="metric-card"))
        
        cards.append(html.Div([
            html.H4(f"{metrics['optimality']:.1%}"),
            html.P("Optimality")
        ], className="metric-card"))
        
        cards.append(html.Div([
            html.H4("✅" if metrics['quantum_advantage'] else "❌"),
            html.P("Quantum Advantage")
        ], className="metric-card"))
        
        return cards
    
    def get_algorithm_info(self, algorithm: str) -> html.Div:
        """Get algorithm information"""
        info = {
            'qaoa': {
                'name': 'Quantum Approximate Optimization Algorithm (QAOA)',
                'description': 'A variational quantum algorithm that uses quantum superposition to explore multiple paths simultaneously.',
                'complexity': 'O(n²) with quantum speedup potential',
                'best_for': 'Multi-objective optimization, complex constraint handling'
            },
            'astar': {
                'name': 'A* Search Algorithm',
                'description': 'A classical graph traversal algorithm that uses heuristics to find optimal paths efficiently.',
                'complexity': 'O(b^d) where b is branching factor and d is depth',
                'best_for': 'Single-objective pathfinding with known heuristics'
            }
        }
        
        algo_info = info.get(algorithm, info['qaoa'])
        
        return html.Div([
            html.H4(algo_info['name']),
            html.P(algo_info['description']),
            html.P(f"Complexity: {algo_info['complexity']}"),
            html.P(f"Best for: {algo_info['best_for']}")
        ])
    
    def run(self, debug=True):
        """Run the dashboard"""
        print(f"🚀 Starting Quantum Robotics Dashboard on http://localhost:{self.port}")
        self.app.run(debug=debug, port=self.port)

# Example usage
if __name__ == "__main__":
    dashboard = QuantumRoboticsDashboard(port=8050)
    dashboard.run()
    
    # To run: python quantum_dashboard.py
    # Then open http://localhost:8050 in your browser