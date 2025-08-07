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
    def __init__(self, grid_size, title="Quantum Pathfinding Visualization", start_image_path=None, goal_image_path=None):
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

    def create_static_plot(self, obstacles=None, path=None, start=None, 
                          goal=None, current_step=None):
        """
        Creates a single static plot showing the grid, obstacles, path, start, goal,
        and optionally highlighting the current position at a specific step.
        Args:
            obstacles (list of tuples): List of (row, col) coordinates for obstacles.
            path (list of tuples): Path in format [(row, col, t), ...].
            start (tuple): (row, col) coordinate of the start.
            goal (tuple): (row, col) coordinate of the goal.
            current_step (int): If provided, highlights the position at this time step.
        Returns:
            plotly.graph_objects.Figure: The Plotly figure object.
        """
        fig = go.Figure()
        # --- 1. Draw Grid Background ---
        # Add grid lines
        # Vertical lines (along columns)
        for c in range(self.cols + 1):
            fig.add_shape(type='line', x0=c-0.5, y0=-0.5, x1=c-0.5, y1=self.rows-0.5,
                          line=dict(color=self.colors['grid_lines'], width=1))
        # Horizontal lines (along rows)
        for r in range(self.rows + 1):
            # Y coordinates for horizontal lines are at row boundaries
            y_plotly = r - 0.5
            fig.add_shape(type='line', x0=-0.5, y0=y_plotly, x1=self.cols-0.5, 
                         y1=y_plotly, line=dict(color=self.colors['grid_lines'], width=1))
        
        # --- 2. Add Obstacles ---
        if obstacles:
            obs_coords_converted = self._convert_coordinates(obstacles)
            if obs_coords_converted:
                 obs_xs, obs_ys = zip(*obs_coords_converted)
                 fig.add_trace(go.Scatter(
                     x=obs_xs, y=obs_ys,
                     mode='markers',
                     marker=dict(color=self.colors['obstacle'], size=self._calculate_marker_size('obstacle'), symbol=self.symbols['obstacle']),
                     name='Obstacles',
                     showlegend=True
                 ))
        
        # --- 3. Add Path ---
        if path:
            path_coords_converted = self._convert_coordinates(path)
            if path_coords_converted:
                 path_xs, path_ys = zip(*path_coords_converted)
                 # Add path line
                 fig.add_trace(go.Scatter(
                     x=path_xs, y=path_ys,
                     mode='lines+markers',
                     line=dict(color=self.colors['path_line'], width=2),
                     marker=dict(color=self.colors['path_marker'], size=self._calculate_marker_size('path_marker'), symbol='circle'),
                     name='Path',
                     showlegend=True
                 ))
                 # Highlight Current Position (if step specified and valid)
                 if current_step is not None and 0 <= current_step < len(path):
                      curr_x, curr_y = path_coords_converted[current_step]
                      # Use goal image for current position if available
                      if self._start_image_base64:
                          self._add_image_marker(fig, curr_x, curr_y, self._start_image_base64, f'Current (Step {current_step})', self.image_marker_size_factor)
                          # Add invisible marker for hover and legend
                          fig.add_trace(go.Scatter(
                              x=[curr_x], y=[curr_y],
                              mode='markers',
                              marker=dict(color='rgba(0,0,0,0)', size=0), # Invisible
                              name=f'Current (Step {current_step})',
                              showlegend=True,
                              hovertemplate=f'Current (Step {current_step})<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>'
                          ))
                      else:
                          # Fallback to standard marker
                          fig.add_trace(go.Scatter(
                              x=[curr_x], y=[curr_y],
                              mode='markers',
                              marker=dict(color=self.colors['current_position'], size=self._calculate_marker_size('current_position'), symbol=self.symbols['current_position']),
                              name=f'Current (Step {current_step})',
                              showlegend=True
                          ))
        
        # --- 4. Add Start Point ---
        if start:
            start_converted = self._convert_coordinates([start])[0]
            start_x, start_y = start_converted
            # Try to add image marker first
            if self._start_image_base64:
                self._add_image_marker(fig, start_x, start_y, self._start_image_base64, 'Start', self.image_marker_size_factor)
                # Add invisible marker for hover and legend
                fig.add_trace(go.Scatter(
                    x=[start_x], y=[start_y],
                    mode='markers',
                    marker=dict(color='rgba(0,0,0,0)', size=0), # Invisible
                    name='Start',
                    showlegend=True,
                    hovertemplate='Start<br>X: %{x}<br>Y: %{y}<extra></extra>'
                ))
            else:
                # Fallback to standard marker
                fig.add_trace(go.Scatter(
                    x=[start_x], y=[start_y],
                    mode='markers',
                    marker=dict(color=self.colors['start'], size=self._calculate_marker_size('goal'), symbol=self.symbols['start']),
                    name='Start',
                    showlegend=True
                ))
        
        # --- 5. Add Goal Point ---
        if goal:
            goal_converted = self._convert_coordinates([goal])[0]
            goal_x, goal_y = goal_converted
            # Try to add image marker first
            if self._goal_image_base64:
                self._add_image_marker(fig, goal_x, goal_y, self._goal_image_base64, 'Goal', self.image_marker_size_factor)
                # Add invisible marker for hover and legend
                fig.add_trace(go.Scatter(
                    x=[goal_x], y=[goal_y],
                    mode='markers',
                    marker=dict(color='rgba(0,0,0,0)', size=0), # Invisible
                    name='Goal',
                    showlegend=True,
                    hovertemplate='Goal<br>X: %{x}<br>Y: %{y}<extra></extra>'
                ))
            else:
                # Fallback to standard marker
                fig.add_trace(go.Scatter(
                    x=[goal_x], y=[goal_y],
                    mode='markers',
                    marker=dict(color=self.colors['goal'], size=self._calculate_marker_size('goal'), symbol=self.symbols['goal']),
                    name='Goal',
                    showlegend=True
                ))
        
        # --- 6. Layout and Axes ---
        # Use consistent sizing - scale up to match step-by-step plot size
        base_width, base_height = self._calculate_figure_size()
        
        # Scale up the static plot to match the step-by-step plot size
        # Use adaptive scaling based on grid size
        scale_factor = self._calculate_static_plot_scale_factor()
        width = int(base_width * scale_factor)
        height = int(base_height * scale_factor)
        
        fig.update_layout(
            title=self.title,
            xaxis=dict(
                range=[-0.5, self.cols - 0.5], # X range based on columns
                showgrid=False, 
                zeroline=False,
                showticklabels=True,
                dtick=1, 
                title='Column'
            ),
            yaxis=dict(
                range=[-0.5, self.rows - 0.5], # Y range based on rows
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                dtick=1,
                title='Row',
                # Reverse the y-axis so that row 0 is at the top (matrix notation)
                autorange='reversed',
                scaleanchor="x", # Keep square grid if desired
                scaleratio=1
            ),
            showlegend=True,
            width=width,
            height=height
        )
        return fig

    def create_step_by_step_plot(self, obstacles, path, start=None, goal=None):
        """
        Creates a subplot visualization showing the path evolution over time steps.
        Args:
            obstacles (list of tuples): List of (row, col) coordinates for obstacles.
            path (list of tuples): Path in format [(row, col, t), ...].
            start (tuple): (row, col) coordinate of the start.
            goal (tuple): (row, col) coordinate of the goal.
        Returns:
            plotly.graph_objects.Figure: The Plotly figure object with subplots.
        """
        if not path:
            raise ValueError("Path is required for step-by-step visualization.")
        num_steps = len(path)
        import math
        cols_subplot = math.ceil(math.sqrt(num_steps))
        rows_subplot = math.ceil(num_steps / cols_subplot)
        
        # Calculate consistent sizing based on grid size (same as static plot)
        base_width, base_height = self._calculate_figure_size()
        
        # Calculate subplot dimensions to maintain consistent cell sizes
        subplot_width = base_width / cols_subplot
        subplot_height = base_height / rows_subplot
        
        fig = make_subplots(
            rows=rows_subplot, cols=cols_subplot,
            subplot_titles=[f"Step {i}" for i in range(num_steps)],
            horizontal_spacing=0.02,  # Consistent spacing
            vertical_spacing=0.08,    # Consistent spacing
            specs=[[{"secondary_y": False} for _ in range(cols_subplot)] for _ in range(rows_subplot)]
        )
        
        # Convert coordinates once
        obstacles_converted = self._convert_coordinates(obstacles) if obstacles else []
        path_converted = self._convert_coordinates(path)
        goal_converted = self._convert_coordinates([goal])[0] if goal else None
        start_converted = self._convert_coordinates([start])[0] if start else None

        for i, (x_plotly, y_plotly) in enumerate(path_converted):
            row_subplot = (i // cols_subplot) + 1
            col_subplot = (i % cols_subplot) + 1
            # Add grid background for subplot
            for c in range(self.cols + 1):
                fig.add_shape(type='line', x0=c-0.5, y0=-0.5, x1=c-0.5, y1=self.rows-0.5,
                              line=dict(color=self.colors['grid_lines'], width=1),
                              row=row_subplot, col=col_subplot)
            for r in range(self.rows + 1):
                y_plotly_grid = r - 0.5
                fig.add_shape(type='line', x0=-0.5, y0=y_plotly_grid, x1=self.cols-0.5, 
                             y1=y_plotly_grid, line=dict(color=self.colors['grid_lines'], width=1),
                             row=row_subplot, col=col_subplot)
            
            # Add obstacles to subplot
            if obstacles_converted:
                obs_xs, obs_ys = zip(*obstacles_converted)
                fig.add_trace(go.Scatter(
                    x=list(obs_xs), y=list(obs_ys),
                    mode='markers',
                    marker=dict(color=self.colors['obstacle'], size=self._calculate_marker_size('obstacle'), symbol=self.symbols['obstacle']),
                    name='Obstacles',
                    showlegend=(i==0)
                ), row=row_subplot, col=col_subplot)
            
            # Add Start (if defined and on first step)
            if start_converted and i == 0:
                # Add start marker (either image or fallback)
                if self._start_image_base64:
                    # Add image marker to this specific subplot
                    fig.add_layout_image(
                        dict(
                            source=self._start_image_base64,
                            x=start_converted[0],
                            y=start_converted[1],
                            xref=f"x{i+1}" if i > 0 else "x",
                            yref=f"y{i+1}" if i > 0 else "y",
                            sizex=self.image_marker_size_factor,
                            sizey=self.image_marker_size_factor,
                            sizing="contain",
                            opacity=1.0,
                            layer="above",
                            xanchor="center",
                            yanchor="middle"
                        )
                    )
                else:
                    # Fallback to standard marker
                    fig.add_trace(go.Scatter(
                        x=[start_converted[0]], y=[start_converted[1]],
                        mode='markers',
                        marker=dict(color=self.colors['start'], size=self._calculate_marker_size('goal'), symbol=self.symbols['start']),
                        name='Start',
                        showlegend=(i==0)
                    ), row=row_subplot, col=col_subplot)
            
            # Add partial path up to current step
            if i >= 0:
                partial_path_coords = path_converted[:i+1]
                pxs, pys = zip(*partial_path_coords)
                fig.add_trace(go.Scatter(
                    x=list(pxs), y=list(pys),
                    mode='lines+markers',
                    line=dict(color=self.colors['path_line'], width=2),
                    marker=dict(color=self.colors['path_marker'], size=self._calculate_marker_size('path_marker')),
                    name='Path',
                    showlegend=(i==0)
                ), row=row_subplot, col=col_subplot)
            
            # Add current position marker (use goal image if available)
            if self._start_image_base64:
                # Use the goal image (Scooby) for current position
                fig.add_layout_image(
                    dict(
                        source=self._start_image_base64,
                        x=x_plotly,
                        y=y_plotly,
                        xref=f"x{i+1}" if i > 0 else "x",
                        yref=f"y{i+1}" if i > 0 else "y",
                        sizex=self.image_marker_size_factor,
                        sizey=self.image_marker_size_factor,
                        sizing="contain",
                        opacity=1.0,
                        layer="above",
                        xanchor="center",
                        yanchor="middle"
                    )
                )
            else:
                # Fallback to standard marker
                fig.add_trace(go.Scatter(
                    x=[x_plotly], y=[y_plotly],
                    mode='markers',
                    marker=dict(color=self.colors['current_position'], size=self._calculate_marker_size('current_position'), symbol=self.symbols['current_position']),
                    name='Current',
                    showlegend=(i==0)
                ), row=row_subplot, col=col_subplot)
            
            # Add Goal (show until reached)
            if goal_converted:
                opacity = 1.0
                current_x_plotly, current_y_plotly = path_converted[i]
                goal_x_plotly, goal_y_plotly = goal_converted
                if current_x_plotly == goal_x_plotly and current_y_plotly == goal_y_plotly:
                    opacity = 0.3  # Fade goal if reached on this step
                
                # Add goal marker (either image or fallback)
                if self._goal_image_base64:
                    # Add image marker to this specific subplot
                    fig.add_layout_image(
                        dict(
                            source=self._goal_image_base64,
                            x=goal_converted[0],
                            y=goal_converted[1],
                            xref=f"x{i+1}" if i > 0 else "x",
                            yref=f"y{i+1}" if i > 0 else "y",
                            sizex=self.image_marker_size_factor,
                            sizey=self.image_marker_size_factor,
                            sizing="contain",
                            opacity=opacity,
                            layer="above",
                            xanchor="center",
                            yanchor="middle"
                        )
                    )
                else:
                    # Fallback to standard marker
                    fig.add_trace(go.Scatter(
                        x=[goal_converted[0]], y=[goal_converted[1]],
                        mode='markers',
                        marker=dict(color=self.colors['goal'], size=self._calculate_marker_size('goal'), symbol=self.symbols['goal'], opacity=opacity),
                        name='Goal',
                        showlegend=(i==0)
                    ), row=row_subplot, col=col_subplot)

            # Update axes for subplot with consistent ranges
            fig.update_xaxes(
                range=[-0.5, self.cols - 0.5], 
                showgrid=False, 
                zeroline=False, 
                dtick=1, 
                row=row_subplot, 
                col=col_subplot,
                title='Column' if row_subplot == rows_subplot else None  # Only show title on bottom row
            )
            # Reverse the y-axis so that row 0 is at the top (matrix notation)
            fig.update_yaxes(
                range=[-0.5, self.rows - 0.5], 
                showgrid=False, 
                zeroline=False, 
                dtick=1, 
                autorange='reversed', 
                row=row_subplot, 
                col=col_subplot,
                title='Row' if col_subplot == 1 else None  # Only show title on leftmost column
            )

        # Note: Start and Goal images are now added to each subplot individually
        # within the loop above, so we don't need global layout images here.

        # Calculate total figure size to maintain consistent cell sizes
        total_width = base_width * cols_subplot
        total_height = base_height * rows_subplot
        
        fig.update_layout(
            title=f"{self.title} - Step by Step",
            showlegend=False,
            width=total_width,
            height=total_height
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

# --- Example Usage ---
if __name__ == "__main__":
    # Example data using (row, col) format
    grid_size = (3, 4) # (rows, cols)
    obstacles = [(1, 1), (2, 3)] 
    path = [(2, 0, 0), (2, 1, 1), (2, 2, 2), (1, 2, 3), (0, 2, 4), (0, 3, 5)]
    start = (2, 0) # row 2, col 0
    goal = (0, 3)  # row 0, col 3

    # --- Create visualizer instance WITH image paths ---
    # Replace 'path/to/start_image.png' and 'path/to/goal_image.svg' with your actual paths
    visualizer = QuantumRoboticsVisualizer(
        grid_size, 
        title="My Quantum Path with Custom Images (Matrix Coords)",
        start_image_path="path/to/your/start_image.png", # e.g., "start.png" or "start.svg"
        goal_image_path="path/to/your/goal_image.svg"     # e.g., "goal.png" or "goal.svg"
    )
    
    # --- Create and Show Static Plot ---
    static_fig = visualizer.create_static_plot(obstacles=obstacles, path=path, start=start, goal=goal, current_step=3)
    # visualizer.show(static_fig) # Uncomment to display in notebook
    # visualizer.write_html(static_fig, "static_path_with_images.html")
    # visualizer.write_image(static_fig, "static_path_with_images.png")

    # --- Create and Show Step-by-Step Plot ---
    step_fig = visualizer.create_step_by_step_plot(obstacles=obstacles, path=path, start=start, goal=goal)
    # visualizer.show(step_fig) # Uncomment to display in notebook
    # visualizer.write_html(step_fig, "step_by_step_path_with_images.html")
    # visualizer.write_image(step_fig, "step_by_step_path_with_images.png", width=900, height=600)