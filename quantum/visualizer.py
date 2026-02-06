# quantum_pathfinding_visualizer_with_images.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import base64
from io import BytesIO
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL (Pillow) not found. Image conversion might be limited.")

class QuantumRoboticsVisualizer:
    """
    Visualizes quantum pathfinding results on a 2D grid using Plotly.
    Designed to work in Jupyter notebooks and can export to HTML for web.
    Handles coordinate system conversion for matrix indexing (row, col) -> (x, y).
    Input format: coordinates are (row, col) tuples.
    Plotly display: x=col, y=row (with Y axis increasing upwards by default).
    Grid display: Configured so (0,0) input is at the top-left visually.

    Enhanced to support custom images for Start and Goal markers.
    """
    def __init__(self, grid_size, title="Quantum Pathfinding Visualization", start_image_path=None, goal_image_path=None, obstacle_image_path=None):
        """
        Initializes the visualizer.
        Args:
            grid_size (tuple): (num_rows, num_cols) of the grid.
            title (str): Title for the plot.
            start_image_path (str, optional): Path to the image file for the Start marker.
            goal_image_path (str, optional): Path to the image file for the Goal marker.
        """
        self.rows, self.cols = grid_size # Store as rows, cols
        self.title = title
        self.start_image_path = start_image_path
        self.goal_image_path = goal_image_path
        self._start_image_base64 = self._load_image_base64(start_image_path)
        self._goal_image_base64 = self._load_image_base64(goal_image_path)
        self._obstacle_image_base64 = self._load_image_base64(obstacle_image_path)

        # --- Default Styling ---
        self.colors = {
            'background': 'white', 
            'grid_lines': 'lightgrey',
            'obstacle': 'black',
            'start': 'blue',
            'goal': 'red',
            'path_line': 'orange',
            'path_marker': 'orange',
            'current_position': 'green'
        }
        self.symbols = {
            'obstacle': 'square',
            'start': 'circle', # Fallback symbol
            'goal': 'diamond', # Fallback symbol
            'current_position': 'star'
        }
        self.sizes = {
            'obstacle': 20,
            'goal': 15, # This will be used for fallback marker size
            'current_position': 15,
            'path_marker': 8
        }
        # Image display size - now relative to grid cell size
        self.image_marker_size_factor = 0.8 # Images will take up 80% of a cell
        
        # Multi-robot palette
        self.robot_colors = [
            'blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'olive'
        ]

    def _load_image_base64(self, image_path):
        """
        Loads an image file and converts it to a base64 string for embedding in Plotly.
        Supports PNG and JPG natively, SVG needs PIL for conversion or special handling.
        """
        if not image_path:
            return None

        try:
            # Try to determine image type from extension
            if image_path.lower().endswith('.svg'):
                # For SVG, we can embed it directly as a data URI string
                with open(image_path, 'rb') as f:
                    svg_data = f.read()
                # Encode the raw SVG data
                encoded_data = base64.b64encode(svg_data).decode()
                return f"data:image/svg+xml;base64,{encoded_data}"
            else:
                # For PNG, JPG, etc.
                if PIL_AVAILABLE:
                    # Use PIL to open and convert to PNG bytes
                    img = Image.open(image_path)
                    img_buffer = BytesIO()
                    # Convert to PNG to ensure compatibility
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    encoded_data = base64.b64encode(img_buffer.read()).decode()
                    return f"data:image/png;base64,{encoded_data}"
                else:
                    # If PIL is not available, try reading raw bytes (works for PNG)
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                    encoded_data = base64.b64encode(image_data).decode()
                    # Guess the type based on extension for the data URI
                    if image_path.lower().endswith('.png'):
                        mime_type = 'image/png'
                    elif image_path.lower().endsendswith(('.jpg', '.jpeg')):
                        mime_type = 'image/jpeg'
                    else:
                        mime_type = 'image/png' # Default guess
                    return f"{mime_type};base64,{encoded_data}"
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}. Using fallback marker.")
            return None

    def _convert_coordinates(self, coords):
        """
        Converts matrix coordinates (row, col) to Plotly coordinates (x, y).
        Plotly X corresponds to column index.
        Plotly Y corresponds to row index, following matrix notation 
        where y increases downward.
        """
        converted = []
        for coord in coords:
            # Input is (row, col) or (row, col, t)
            row, col = coord[0], coord[1] 
            # Plotly X is column index
            x_plotly = col
            # Plotly Y is row index, following matrix notation
            # Row 0 should be at the top (y=0), row increases downward
            y_plotly = row
            converted.append((x_plotly, y_plotly))
        return converted

    def _calculate_figure_size(self):
        """
        Calculates consistent figure size based on grid dimensions.
        Returns tuple of (width, height) in pixels.
        """
        base_width = max(400, self.cols * 50)
        base_height = max(400, self.rows * 50)
        return base_width, base_height

    def _calculate_cell_size(self):
        """
        Calculates the size of a single grid cell in pixels.
        This helps ensure consistent sizing across different plot types.
        """
        width, height = self._calculate_figure_size()
        cell_width = width / self.cols
        cell_height = height / self.rows
        return min(cell_width, cell_height)  # Use the smaller dimension to maintain square cells

    def _calculate_marker_size(self, marker_type):
        """
        Calculates consistent marker sizes based on grid cell size.
        Args:
            marker_type (str): Type of marker ('obstacle', 'goal', 'current_position', 'path_marker')
        Returns:
            int: Marker size in pixels
        """
        cell_size = self._calculate_cell_size()
        base_sizes = {
            'obstacle': 20,
            'goal': 15,
            'current_position': 15,
            'path_marker': 8
        }
        
        # Scale marker size based on cell size for consistency
        base_size = base_sizes.get(marker_type, 10)
        scale_factor = cell_size / 50  # Normalize to the base cell size
        return max(5, int(base_size * scale_factor))  # Ensure minimum size

    def _calculate_static_plot_scale_factor(self):
        """
        Calculates an appropriate scale factor for the static plot to match step-by-step plot size.
        Returns:
            float: Scale factor to apply to base figure size
        """
        # For larger grids, we want larger plots
        grid_area = self.rows * self.cols
        
        if grid_area <= 9:  # Small grids (3x3 or smaller)
            return 1.5
        elif grid_area <= 16:  # Medium grids (4x4)
            return 2
        elif grid_area <= 25:  # Larger grids (5x5)
            return 2.5
        else:  # Very large grids
            return 3

    def _add_image_marker(self, fig, x, y, image_base64_data, name, size_factor=0.8):
        """
        Adds an image as a marker to the figure at the specified coordinates.
        The image is centered at (x,y) and sized to fit within a grid cell.
        """
        if image_base64_data:
            # Use consistent image size relative to grid cell size
            image_size = size_factor  # Since grid cells are unit size (1x1) in plot coordinates
            
            # Add the image as a layout image, centered at (x,y)
            fig.add_layout_image(
                dict(
                    source=image_base64_data,
                    x=x,  # Center x
                    y=y,  # Center y
                    xref="x",
                    yref="y",
                    sizex=image_size,
                    sizey=image_size,
                    sizing="contain",  # How the image fits within sizex/sizey
                    opacity=1.0,
                    layer="above",  # Place above traces
                    xanchor="center",  # Anchor point for x positioning
                    yanchor="middle"   # Anchor point for y positioning
                )
            )

    def create_static_plot(self, obstacles=None, path=None, start=None, goal=None, current_step=None,
                        problem=None, robot_paths=None):
        """
        Creates a single static plot with support for multiple robots.
        Args:
            obstacles: List of obstacle coordinates
            path: (Legacy) Single robot path.
            start: (Legacy) Single start.
            goal: (Legacy) Single goal.
            current_step: Current timestep to show position for.
            problem: Problem instance (needed for multi-robot start/goals)
            robot_paths: Dictionary {robot_id: path_list}
        """
        # --- Normalize Inputs for Multi-Robot ---
        robots_data = {}
        
        # 1. Paths
        if robot_paths:
            for r_idx, (r_key, r_path) in enumerate(robot_paths.items()):
                robots_data[r_idx] = {'path': r_path, 'name': str(r_key)}
        elif path:
            robots_data[0] = {'path': path, 'name': 'Robot'}
            
        # 2. Stats/Goals/Colors
        problem_robots_list = list(problem.robots.values()) if (problem and hasattr(problem, 'robots')) else []
        
        for idx in range(len(robots_data)):
            data = robots_data[idx]
            # Color
            data['color'] = self.robot_colors[idx % len(self.robot_colors)]
            
            # Start/Goal
            if problem and idx < len(problem_robots_list):
                robot_obj = problem_robots_list[idx]
                data['start'] = robot_obj.start
                data['goal'] = robot_obj.goal
            else:
                if idx == 0:
                    data['start'] = start if start is not None else getattr(problem, "start", None)
                    data['goal'] = goal if goal is not None else getattr(problem, "end", None)
            
            # Current Position
            data['current'] = None
            if current_step is not None and data['path']:
                if 0 <= current_step < len(data['path']):
                    data['current'] = data['path'][current_step]

        fig = go.Figure()

        # --- Terrain Background ---
        if problem is not None and hasattr(problem, 'grid') and problem.grid.terrain is not None:
            for r in range(self.rows):
                for c in range(self.cols):
                    mat_index = problem.grid.get_terrain_at(r, c)
                    if mat_index is not None:
                        color = problem.grid.get_color(mat_index)
                        fig.add_shape(type="rect", x0=c-0.5, y0=r-0.5, x1=c+0.5, y1=r+0.5,
                                    fillcolor=color, line=dict(width=0), layer="below")

        # --- Grid Lines ---
        for c in range(self.cols + 1):
            fig.add_shape(type='line', x0=c-0.5, y0=-0.5, x1=c-0.5, y1=self.rows-0.5,
                        line=dict(color=self.colors['grid_lines'], width=1))
        for r in range(self.rows + 1):
            fig.add_shape(type='line', x0=-0.5, y0=r-0.5, x1=self.cols-0.5, y1=r-0.5,
                        line=dict(color=self.colors['grid_lines'], width=1))

        # --- Obstacles ---
        obstacles_converted = self._convert_coordinates(obstacles) if obstacles else []
        if obstacles_converted:
            obs_xs, obs_ys = zip(*obstacles_converted)
            hover_text_obs = [f"Obstacle<br>X: {int(x)}<br>Y: {int(y)}" for x, y in zip(obs_xs, obs_ys)]
            
            if self._obstacle_image_base64:
                for x, y, text in zip(obs_xs, obs_ys, hover_text_obs):
                    self._add_image_marker(fig, x, y, self._obstacle_image_base64, 'Obstacle', self.image_marker_size_factor)
                fig.add_trace(go.Scatter(x=obs_xs, y=obs_ys, mode='markers', marker=dict(color='rgba(0,0,0,0)', size=0),
                                       name='Obstacles', showlegend=True, hovertemplate='%{text}<extra></extra>', text=hover_text_obs))
            else:
                fig.add_trace(go.Scatter(x=obs_xs, y=obs_ys, mode='markers',
                                       marker=dict(color=self.colors['obstacle'], size=self._calculate_marker_size('obstacle'), symbol=self.symbols['obstacle']),
                                       name='Obstacles', showlegend=True, hovertemplate='%{text}<extra></extra>', text=hover_text_obs))

        # --- Plot Robots ---
        for idx, data in robots_data.items():
            color = data['color']
            name = data['name']
            
            path_conv = self._convert_coordinates(data['path']) if data['path'] else []
            start_conv = self._convert_coordinates([data['start']])[0] if data.get('start') else None
            goal_conv = self._convert_coordinates([data['goal']])[0] if data.get('goal') else None
            curr_conv = self._convert_coordinates([data['current']])[0] if data.get('current') else None
            
            # 1. Path
            if path_conv and current_step is not None:
                partial = path_conv[:current_step + 1]
                if len(partial) > 1:
                    pxs, pys = zip(*partial)
                    fig.add_trace(go.Scatter(x=pxs, y=pys, mode='lines', line=dict(color=color, width=3),
                                           name=f'{name} Path', showlegend=True, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=pxs, y=pys, mode='markers', marker=dict(color=color, size=6, symbol='circle'),
                                           showlegend=False, hovertemplate=f'{name} Path<br>Step %{{id}}<extra></extra>', ids=list(range(len(pxs)))))

            # 2. Start
            if start_conv:
                # Use S text marker for clarity in multi-robot
                 fig.add_trace(go.Scatter(x=[start_conv[0]], y=[start_conv[1]], mode='markers+text',
                                        marker=dict(color=color, size=self._calculate_marker_size('goal'), symbol='circle-open', line=dict(width=2)),
                                        text=['S'], textfont=dict(color=color), name=f'{name} Start', showlegend=False,
                                        hovertemplate=f'{name} Start<extra></extra>'))

            # 3. Goal
            if goal_conv:
                opacity = 1.0
                if curr_conv and curr_conv == goal_conv:
                    opacity = 0.3
                fig.add_trace(go.Scatter(x=[goal_conv[0]], y=[goal_conv[1]], mode='markers',
                                       marker=dict(color=color, size=self._calculate_marker_size('goal'), symbol='diamond', opacity=opacity),
                                       name=f'{name} Goal', showlegend=False, hovertemplate=f'{name} Goal<extra></extra>'))

            # 4. Current Position
            if curr_conv:
                use_image = (idx == 0 and self._start_image_base64)
                if use_image:
                     self._add_image_marker(fig, curr_conv[0], curr_conv[1], self._start_image_base64, name, self.image_marker_size_factor)
                else:
                    fig.add_trace(go.Scatter(x=[curr_conv[0]], y=[curr_conv[1]], mode='markers',
                                           marker=dict(color=color, size=self._calculate_marker_size('current_position'), symbol='star'),
                                           name=name, showlegend=True, hovertemplate=f'{name} (Current)<extra></extra>'))

        # --- Final Layout ---
        fig.update_layout(
            title=self.title,
            xaxis=dict(range=[-0.5, self.cols - 0.5], showgrid=False, zeroline=False, dtick=1, title="Column", scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[self.rows - 0.5, -0.5], showgrid=False, zeroline=False, dtick=1, title="Row"),
            showlegend=True,
            plot_bgcolor=self.colors['background'],
            margin=dict(l=50, r=50, t=80, b=50),
            width=600, height=600 * self.rows / max(self.cols, self.rows) * (self.cols / self.rows if self.cols > self.rows else 1),
            hovermode='closest'
        )
        return fig

    def create_step_by_step_plot(self, obstacles, path=None, start=None, goal=None, problem=None, robot_paths=None):
        """
        Creates a subplot visualization showing the path evolution over time steps.
        Now includes terrain background from problem.grid, just like static plot.
        """
        # --- Normalize Inputs ---
        robots_data = {}
        if robot_paths:
            for r_idx, (r_key, r_path) in enumerate(robot_paths.items()):
                robots_data[r_idx] = {'path': r_path, 'name': str(r_key)}
        elif path:
            robots_data[0] = {'path': path, 'name': 'Robot'}
            
        if not robots_data:
             raise ValueError("Path is required for step-by-step visualization.")
             
        # Determine max steps
        max_steps = 0
        for data in robots_data.values():
            if data['path']:
                max_steps = max(max_steps, len(data['path']))

        # Determine Starts/Goals/Colors
        problem_robots_list = list(problem.robots.values()) if (problem and hasattr(problem, 'robots')) else []
        for idx, data in robots_data.items():
            data['color'] = self.robot_colors[idx % len(self.robot_colors)]
            if problem and idx < len(problem_robots_list):
                 data['start'] = problem_robots_list[idx].start
                 data['goal'] = problem_robots_list[idx].goal
            else:
                 if idx == 0:
                     data['start'] = start
                     data['goal'] = goal
                     
            # Pre-convert coordinates to avoid re-doing it in loop
            data['path_conv'] = self._convert_coordinates(data['path']) if data['path'] else []
            data['start_conv'] = self._convert_coordinates([data['start']])[0] if data.get('start') else None
            data['goal_conv'] = self._convert_coordinates([data['goal']])[0] if data.get('goal') else None

        import math
        cols_subplot = math.ceil(math.sqrt(max_steps))
        rows_subplot = math.ceil(max_steps / cols_subplot)

        # Calculate consistent sizing based on grid size
        base_width, base_height = self._calculate_figure_size()

        fig = make_subplots(
            rows=rows_subplot, cols=cols_subplot,
            subplot_titles=[f"Step {i}" for i in range(max_steps)],
            horizontal_spacing=0.02,
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False} for _ in range(cols_subplot)] for _ in range(rows_subplot)]
        )

        # Convert obstacles once
        obstacles_converted = self._convert_coordinates(obstacles) if obstacles else []

        # Loop through each step
        for i in range(max_steps):
            row_subplot = (i // cols_subplot) + 1
            col_subplot = (i % cols_subplot) + 1
            
            subplot_index = (row_subplot - 1) * cols_subplot + (col_subplot - 1) + 1
            xref = f"x{subplot_index}" if subplot_index > 1 else "x"
            yref = f"y{subplot_index}" if subplot_index > 1 else "y"

            # 1. Terrain
            if problem is not None and hasattr(problem, 'grid') and problem.grid.terrain is not None:
                for r in range(self.rows):
                    for c in range(self.cols):
                        mat_index = problem.grid.get_terrain_at(r, c)
                        if mat_index is not None:
                            color = problem.grid.get_color(mat_index)
                            fig.add_shape(type="rect", x0=c-0.5, y0=r-0.5, x1=c+0.5, y1=r+0.5,
                                        fillcolor=color, line=dict(width=0), layer="below",
                                        row=row_subplot, col=col_subplot)

            # 2. Grid Lines
            for c in range(self.cols + 1):
                fig.add_shape(type='line', x0=c-0.5, y0=-0.5, x1=c-0.5, y1=self.rows-0.5,
                            line=dict(color=self.colors['grid_lines'], width=1), row=row_subplot, col=col_subplot)
            for r in range(self.rows + 1):
                fig.add_shape(type='line', x0=-0.5, y0=r-0.5, x1=self.cols-0.5, y1=r-0.5,
                            line=dict(color=self.colors['grid_lines'], width=1), row=row_subplot, col=col_subplot)

            # 3. Obstacles
            if obstacles_converted:
                obs_xs, obs_ys = zip(*obstacles_converted)
                if self._obstacle_image_base64:
                    for ox, oy in zip(obs_xs, obs_ys):
                        fig.add_layout_image(dict(source=self._obstacle_image_base64, x=ox, y=oy, xref=xref, yref=yref,
                                                sizex=self.image_marker_size_factor, sizey=self.image_marker_size_factor,
                                                sizing="contain", opacity=1.0, layer="above", xanchor="center", yanchor="middle"))
                else:
                    fig.add_trace(go.Scatter(x=list(obs_xs), y=list(obs_ys), mode='markers',
                                           marker=dict(color=self.colors['obstacle'], size=self._calculate_marker_size('obstacle'), symbol='square'),
                                           showlegend=(i==0), name='Obstacles'), row=row_subplot, col=col_subplot)

            # 4. Robots
            for idx, data in robots_data.items():
                color = data['color']
                name = data['name']
                path_conv = data['path_conv']
                start_conv = data['start_conv']
                goal_conv = data['goal_conv']
                
                # Start
                if start_conv and i == 0:
                     fig.add_trace(go.Scatter(x=[start_conv[0]], y=[start_conv[1]], mode='markers',
                                            marker=dict(color=color, size=6, symbol='circle-open', line=dict(width=2)),
                                            showlegend=False), row=row_subplot, col=col_subplot)

                # Goal
                if goal_conv:
                    opacity = 1.0
                    # Check if reached at this step
                    curr_pos = path_conv[i] if i < len(path_conv) else path_conv[-1] if path_conv else None
                    if curr_pos == goal_conv:
                        opacity = 0.3
                    
                    fig.add_trace(go.Scatter(x=[goal_conv[0]], y=[goal_conv[1]], mode='markers',
                                           marker=dict(color=color, size=8, symbol='diamond', opacity=opacity),
                                           showlegend=False), row=row_subplot, col=col_subplot)
                
                # Path history
                if path_conv:
                     # Show path up to step i
                     limit = min(i + 1, len(path_conv))
                     partial = path_conv[:limit]
                     if len(partial) > 1:
                         pxs, pys = zip(*partial)
                         fig.add_trace(go.Scatter(x=pxs, y=pys, mode='lines', line=dict(color=color, width=2),
                                                showlegend=(i==0), name=name), row=row_subplot, col=col_subplot)
                     
                     # Current Position
                     if i < len(path_conv):
                         curr = path_conv[i]
                         # Use scooby for first robot
                         if idx == 0 and self._start_image_base64:
                             fig.add_layout_image(dict(source=self._start_image_base64, x=curr[0], y=curr[1],
                                                     xref=xref, yref=yref, sizex=self.image_marker_size_factor, sizey=self.image_marker_size_factor,
                                                     sizing="contain", opacity=1.0, layer="above", xanchor="center", yanchor="middle"))
                         else:
                             fig.add_trace(go.Scatter(x=[curr[0]], y=[curr[1]], mode='markers',
                                                    marker=dict(color=color, size=10, symbol='star'),
                                                    showlegend=False), row=row_subplot, col=col_subplot)

            # Axis updates
            fig.update_xaxes(range=[-0.5, self.cols-0.5], showgrid=False, zeroline=False, dtick=1, row=row_subplot, col=col_subplot)
            fig.update_yaxes(range=[self.rows-0.5, -0.5], showgrid=False, zeroline=False, dtick=1, row=row_subplot, col=col_subplot)

        total_width = base_width * cols_subplot
        total_height = base_height * rows_subplot
        fig.update_layout(title=f"{self.title} - Step by Step", showlegend=True, width=total_width, height=total_height,
                        margin=dict(l=60, r=60, t=100, b=60))
        return fig

    def show(self, fig):
        """Displays the figure."""
        fig.show()

    def write_html(self, fig, filename):
        """Saves the figure as HTML."""
        fig.write_html(filename)
        print(f"Plot saved to {filename}")

    def write_image(self, fig, filename, format='png', width=None, height=None):
        """Saves the figure as a static image."""
        try:
            fig.write_image(filename, format=format, width=width, height=height)
            print(f"Image saved to {filename}")
        except Exception as e:
             print(f"Failed to save image: {e}. Ensure kaleido (`pip install kaleido`) is installed.")