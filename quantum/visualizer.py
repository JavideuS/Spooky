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
                        problem=None):
        """
        Creates a single static plot with:
        - Grid, obstacles, partial path, current position (with Scooby image),
        - Start shown only if not replaced by current pos,
        - Goal faded if reached,
        - No duplicate Scooby images.
        - Proper hover and legend for all elements.
        """
        if problem is not None:
            if start is None:
                start = getattr(problem, "start", None)
            if goal is None:
                goal = getattr(problem, "end", None)

        fig = go.Figure()

        # --- Terrain Background (if available) ---
        if problem is not None and hasattr(problem, 'grid') and problem.grid.materials is not None:
            for r in range(self.rows):
                for c in range(self.cols):
                    mat_index = problem.grid.get_terrain_at(r, c)
                    color = problem.grid.get_color(mat_index)
                    fig.add_shape(
                        type="rect",
                        x0=c - 0.5, y0=r - 0.5,
                        x1=c + 0.5, y1=r + 0.5,
                        fillcolor=color,
                        line=dict(width=0),
                        layer="below"
                    )

        # --- Grid Lines ---
        for c in range(self.cols + 1):
            fig.add_shape(type='line', x0=c - 0.5, y0=-0.5, x1=c - 0.5, y1=self.rows - 0.5,
                        line=dict(color=self.colors['grid_lines'], width=1))
        for r in range(self.rows + 1):
            y_plotly = r - 0.5
            fig.add_shape(type='line', x0=-0.5, y0=y_plotly, x1=self.cols - 0.5, y1=y_plotly,
                        line=dict(color=self.colors['grid_lines'], width=1))

        # Convert coordinates
        obstacles_converted = self._convert_coordinates(obstacles) if obstacles else []
        path_converted = self._convert_coordinates(path) if path else []
        start_converted = self._convert_coordinates([start])[0] if start else None
        goal_converted = self._convert_coordinates([goal])[0] if goal else None

        # Determine current position
        current_pos = None
        if current_step is not None and path_converted and 0 <= current_step < len(path_converted):
            current_pos = path_converted[current_step]

        # --- Add Obstacles ---
        if obstacles_converted:
            obs_xs, obs_ys = zip(*obstacles_converted)
            hover_texts = [f"Obstacle<br>X: {int(x)}<br>Y: {int(y)}" for x, y in zip(obs_xs, obs_ys)]

            if self._obstacle_image_base64:
                for x, y, text in zip(obs_xs, obs_ys, hover_texts):
                    fig.add_layout_image(
                        dict(
                            source=self._obstacle_image_base64,
                            x=x, y=y,
                            xref="x", yref="y",
                            sizex=self.image_marker_size_factor,
                            sizey=self.image_marker_size_factor,
                            sizing="contain",
                            opacity=1.0,
                            layer="above",
                            xanchor="center", yanchor="middle"
                        )
                    )
                # Single legend entry + hover via dummy trace
                fig.add_trace(go.Scatter(
                    x=obs_xs, y=obs_ys,
                    mode='markers',
                    marker=dict(color='rgba(0,0,0,0)', size=0),
                    name='Obstacles',
                    showlegend=True,
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=obs_xs, y=obs_ys,
                    mode='markers',
                    marker=dict(color=self.colors['obstacle'], size=self._calculate_marker_size('obstacle'), symbol=self.symbols['obstacle']),
                    name='Obstacles',
                    showlegend=True,
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts
                ))

        # --- Add Path Traveled (up to current_step) ---
        if path_converted and current_step is not None:
            partial_path = path_converted[:current_step + 1]
            pxs, pys = zip(*partial_path)

            # Line segment hover: generic label
            hover_line_text = ["Path Segment"] * (len(pxs) - 1) if len(pxs) > 1 else []

            # Marker hover: show position
            hover_marker_text = [f"Path Point<br>X: {int(x)}<br>Y: {int(y)}" for x, y in zip(pxs, pys)]

            # Add path line with hover
            if len(pxs) > 1:
                fig.add_trace(go.Scatter(
                    x=pxs, y=pys,
                    mode='lines',
                    line=dict(color=self.colors['path_line'], width=3),
                    name='Path',
                    showlegend=True,
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_line_text
                ))

            # Add path markers with hover
            fig.add_trace(go.Scatter(
                x=pxs, y=pys,
                mode='markers',
                marker=dict(color=self.colors['path_marker'], size=self._calculate_marker_size('path_marker'), symbol='circle'),
                name='Path',
                showlegend=False,  # Only show "Path" once in legend
                hovertemplate='%{text}<extra></extra>',
                text=hover_marker_text
            ))

        # --- Add Start Marker (only if not replaced by current position) ---
        if start_converted:
            # Only show start marker if current position is NOT on start
            if current_pos is None or current_pos != start_converted:
                if self._start_image_base64:
                    # Show a small indicator or text, NOT the robot image
                    fig.add_trace(go.Scatter(
                        x=[start_converted[0]], y=[start_converted[1]],
                        mode='text',
                        text=["S"],
                        textfont=dict(color="blue", size=14),
                        name='Start',
                        showlegend=True,
                        hovertemplate='Start Position<br>X: %{x}<br>Y: %{y}<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[start_converted[0]], y=[start_converted[1]],
                        mode='markers',
                        marker=dict(color=self.colors['start'], size=self._calculate_marker_size('goal'), symbol='circle-open', line=dict(width=2)),
                        name='Start',
                        showlegend=True,
                        hovertemplate='Start Position<br>X: %{x}<br>Y: %{y}<extra></extra>'
                    ))

        # --- Add Current Position (Always show Scooby image if available) ---
        if current_pos is not None:
            x_curr, y_curr = current_pos
            hover_text = f"Robot (Current)<br>X: {int(x_curr)}<br>Y: {int(y_curr)}"

            if self._start_image_base64:  # Scooby image
                self._add_image_marker(fig, x_curr, y_curr, self._start_image_base64, 'Current', self.image_marker_size_factor)
                fig.add_trace(go.Scatter(
                    x=[x_curr], y=[y_curr],
                    mode='markers',
                    marker=dict(color='rgba(0,0,0,0)', size=0),
                    name='Current Position',
                    showlegend=True,
                    hovertemplate=hover_text + '<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[x_curr], y=[y_curr],
                    mode='markers',
                    marker=dict(color=self.colors['current_position'], size=self._calculate_marker_size('current_position'), symbol=self.symbols['current_position']),
                    name='Current Position',
                    showlegend=True,
                    hovertemplate=hover_text + '<extra></extra>'
                ))

        # --- Add Goal (faded if reached) ---
        if goal_converted:
            show_goal = True
            opacity = 1.0
            if current_pos and current_pos == goal_converted:
                opacity = 0.3
                show_goal = False  # Hide if robot is on goal

            if show_goal or opacity < 1.0:
                if self._goal_image_base64 and opacity > 0:
                    fig.add_layout_image(
                        dict(
                            source=self._goal_image_base64,
                            x=goal_converted[0], y=goal_converted[1],
                            xref="x", yref="y",
                            sizex=self.image_marker_size_factor,
                            sizey=self.image_marker_size_factor,
                            sizing="contain",
                            opacity=opacity,
                            layer="above",
                            xanchor="center", yanchor="middle"
                        )
                    )
                    fig.add_trace(go.Scatter(
                        x=[goal_converted[0]], y=[goal_converted[1]],
                        mode='markers',
                        marker=dict(color='rgba(0,0,0,0)', size=0),
                        name='Goal',
                        showlegend=True,
                        hovertemplate='Goal<br>X: %{x}<br>Y: %{y}<extra></extra>'
                    ))
                elif opacity > 0:
                    fig.add_trace(go.Scatter(
                        x=[goal_converted[0]], y=[goal_converted[1]],
                        mode='markers',
                        marker=dict(color=self.colors['goal'], size=self._calculate_marker_size('goal'), symbol=self.symbols['goal'], opacity=opacity),
                        name='Goal',
                        showlegend=True,
                        hovertemplate='Goal<br>X: %{x}<br>Y: %{y}<extra></extra>'
                    ))

        # --- Final Layout: Tight & Square ---
        fig.update_layout(
            title=self.title,
            xaxis=dict(
                range=[-0.5, self.cols - 0.5],
                showgrid=False,
                zeroline=False,
                dtick=1,
                title="Column",
                scaleanchor="y",
                scaleratio=1,
                constrain="domain"
            ),
            yaxis=dict(
                range=[self.rows - 0.5, -0.5],  # Reversed: top=0
                showgrid=False,
                zeroline=False,
                dtick=1,
                title="Row",
                constrain="domain",
                scaleratio=1
            ),
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50),
            width=600,
            height=600 * self.rows / max(self.cols, self.rows) * (self.cols / self.rows if self.cols > self.rows else 1),
            plot_bgcolor=self.colors['background'],
            hovermode='closest'
        )

        return fig

    def create_step_by_step_plot(self, obstacles, path, start=None, goal=None, problem=None):
        """
        Creates a subplot visualization showing the path evolution over time steps.
        Now includes terrain background from problem.grid, just like static plot.
        """
        if not path:
            raise ValueError("Path is required for step-by-step visualization.")
        num_steps = len(path)
        import math
        cols_subplot = math.ceil(math.sqrt(num_steps))
        rows_subplot = math.ceil(num_steps / cols_subplot)

        # Calculate consistent sizing based on grid size
        base_width, base_height = self._calculate_figure_size()

        fig = make_subplots(
            rows=rows_subplot, cols=cols_subplot,
            subplot_titles=[f"Step {i}" for i in range(num_steps)],
            horizontal_spacing=0.02,
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False} for _ in range(cols_subplot)] for _ in range(rows_subplot)]
        )

        # Convert coordinates once
        obstacles_converted = self._convert_coordinates(obstacles) if obstacles else []
        path_converted = self._convert_coordinates(path)
        goal_converted = self._convert_coordinates([goal])[0] if goal else None
        start_converted = self._convert_coordinates([start])[0] if start else None

        # --- Add Terrain and Grid to Each Subplot ---
        for i, (x_plotly, y_plotly) in enumerate(path_converted):
            row_subplot = (i // cols_subplot) + 1
            col_subplot = (i % cols_subplot) + 1

            subplot_index = (row_subplot - 1) * cols_subplot + (col_subplot - 1) + 1
            xref = f"x{subplot_index}" if subplot_index > 1 else "x"
            yref = f"y{subplot_index}" if subplot_index > 1 else "y"

            # --- Terrain Background (if problem and grid available) ---
            if problem is not None and hasattr(problem, 'grid') and problem.grid.materials is not None:
                for r in range(self.rows):
                    for c in range(self.cols):
                        mat_index = problem.grid.get_terrain_at(r, c)
                        color = problem.grid.get_color(mat_index)
                        fig.add_shape(
                            type="rect",
                            x0=c - 0.5, y0=r - 0.5,
                            x1=c + 0.5, y1=r + 0.5,
                            fillcolor=color,
                            line=dict(width=0),
                            layer="below",
                            row=row_subplot, col=col_subplot
                        )

            # --- Grid Lines ---
            for c in range(self.cols + 1):
                fig.add_shape(
                    type='line', x0=c - 0.5, y0=-0.5, x1=c - 0.5, y1=self.rows - 0.5,
                    line=dict(color=self.colors['grid_lines'], width=1),
                    row=row_subplot, col=col_subplot
                )
            for r in range(self.rows + 1):
                y_plotly_grid = r - 0.5
                fig.add_shape(
                    type='line', x0=-0.5, y0=y_plotly_grid, x1=self.cols - 0.5, y1=y_plotly_grid,
                    line=dict(color=self.colors['grid_lines'], width=1),
                    row=row_subplot, col=col_subplot
                )

            # --- Obstacles ---
            if obstacles_converted:
                obs_xs, obs_ys = zip(*obstacles_converted)
                if self._obstacle_image_base64:
                    for ox, oy in zip(obs_xs, obs_ys):
                        fig.add_layout_image(
                            dict(
                                source=self._obstacle_image_base64,
                                x=ox, y=oy,
                                xref=xref, yref=yref,
                                sizex=self.image_marker_size_factor,
                                sizey=self.image_marker_size_factor,
                                sizing="contain",
                                opacity=1.0,
                                layer="above",
                                xanchor="center", yanchor="middle"
                            )
                        )
                    # Legend entry (once)
                    if i == 0:
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(color='rgba(0,0,0,0)', size=0),
                            name='Obstacles',
                            showlegend=True
                        ), row=row_subplot, col=col_subplot)
                else:
                    fig.add_trace(go.Scatter(
                        x=list(obs_xs), y=list(obs_ys),
                        mode='markers',
                        marker=dict(color=self.colors['obstacle'], size=self._calculate_marker_size('obstacle'), symbol=self.symbols['obstacle']),
                        name='Obstacles',
                        showlegend=(i == 0)
                    ), row=row_subplot, col=col_subplot)

            # --- Start (only on first step) ---
            if start_converted and i == 0:
                if self._start_image_base64:
                    fig.add_layout_image(
                        dict(
                            source=self._start_image_base64,
                            x=start_converted[0], y=start_converted[1],
                            xref=xref, yref=yref,
                            sizex=self.image_marker_size_factor,
                            sizey=self.image_marker_size_factor,
                            sizing="contain",
                            opacity=1.0,
                            layer="above",
                            xanchor="center", yanchor="middle"
                        )
                    )
                else:
                    fig.add_trace(go.Scatter(
                        x=[start_converted[0]], y=[start_converted[1]],
                        mode='markers',
                        marker=dict(color=self.colors['start'], size=self._calculate_marker_size('goal'), symbol=self.symbols['start']),
                        name='Start',
                        showlegend=True
                    ), row=row_subplot, col=col_subplot)

            # --- Partial Path ---
            partial_path = path_converted[:i+1]
            if len(partial_path) > 1:
                pxs, pys = zip(*partial_path)
                fig.add_trace(go.Scatter(
                    x=list(pxs), y=list(pys),
                    mode='lines',
                    line=dict(color=self.colors['path_line'], width=2),
                    name='Path',
                    showlegend=(i == 0)
                ), row=row_subplot, col=col_subplot)
            # Path markers
            fig.add_trace(go.Scatter(
                x=[p[0] for p in partial_path], y=[p[1] for p in partial_path],
                mode='markers',
                marker=dict(color=self.colors['path_marker'], size=self._calculate_marker_size('path_marker')),
                showlegend=False,
                hoverinfo='none'
            ), row=row_subplot, col=col_subplot)

            # --- Current Position (Scooby image) ---
            if self._start_image_base64:
                fig.add_layout_image(
                    dict(
                        source=self._start_image_base64,
                        x=x_plotly, y=y_plotly,
                        xref=xref, yref=yref,
                        sizex=self.image_marker_size_factor,
                        sizey=self.image_marker_size_factor,
                        sizing="contain",
                        opacity=1.0,
                        layer="above",
                        xanchor="center", yanchor="middle"
                    )
                )
            else:
                fig.add_trace(go.Scatter(
                    x=[x_plotly], y=[y_plotly],
                    mode='markers',
                    marker=dict(color=self.colors['current_position'], size=self._calculate_marker_size('current_position'), symbol=self.symbols['current_position']),
                    name='Current',
                    showlegend=(i == 0)
                ), row=row_subplot, col=col_subplot)

            # --- Goal (fades when reached) ---
            if goal_converted:
                opacity = 1.0
                if x_plotly == goal_converted[0] and y_plotly == goal_converted[1]:
                    opacity = 0.3

                if self._goal_image_base64:
                    fig.add_layout_image(
                        dict(
                            source=self._goal_image_base64,
                            x=goal_converted[0], y=goal_converted[1],
                            xref=xref, yref=yref,
                            sizex=self.image_marker_size_factor,
                            sizey=self.image_marker_size_factor,
                            sizing="contain",
                            opacity=opacity,
                            layer="above",
                            xanchor="center", yanchor="middle"
                        )
                    )
                else:
                    fig.add_trace(go.Scatter(
                        x=[goal_converted[0]], y=[goal_converted[1]],
                        mode='markers',
                        marker=dict(color=self.colors['goal'], size=self._calculate_marker_size('goal'), symbol=self.symbols['goal'], opacity=opacity),
                        name='Goal',
                        showlegend=(i == 0)
                    ), row=row_subplot, col=col_subplot)

            # --- Axis Updates ---
            fig.update_xaxes(
                range=[-0.5, self.cols - 0.5],
                showgrid=False, zeroline=False, dtick=1,
                title='Column' if row_subplot == rows_subplot else None,
                row=row_subplot, col=col_subplot
            )
            fig.update_yaxes(
                range=[self.rows - 0.5, -0.5],  # Reversed
                showgrid=False, zeroline=False, dtick=1,
                title='Row' if col_subplot == 1 else None,
                autorange=False,
                row=row_subplot, col=col_subplot
            )

        # --- Final Layout ---
        total_width = base_width * cols_subplot
        total_height = base_height * rows_subplot

        fig.update_layout(
            title=f"{self.title} - Step by Step",
            showlegend=True,
            width=total_width,
            height=total_height,
            margin=dict(l=60, r=60, t=100, b=60)
        )

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