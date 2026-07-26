# quantum_pathfinding_visualizer_with_images.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import base64
from io import BytesIO
from pathlib import Path

from quantum.utils.coordinates import to_robotics_xy
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
    Input format: coordinates are (row, col) or (row, col, t) tuples. When the
    timestep is present it is authoritative — robots with delayed starts
    (start_time > 0) or early finishes are aligned on a shared timeline, the
    same semantics the benchmark validator uses.

    Display conventions (see quantum/utils/coordinates.py):
      - "matrix" (default): axes labeled Row/Column, (0,0) at the top-left,
        row increasing downward.
      - "robotics": axes labeled X/Y, Cartesian Y-up with the origin at the
        bottom-left. The scene is geometrically identical; only the axis
        labeling/orientation changes.

    Multi-robot rendering uses "subway-style" lanes: each robot's path is drawn
    with a small per-robot offset inside the cell, so paths sharing cells run as
    thin parallel tracks instead of overlapping. Actual conflicts (two robots in
    the same cell at the same timestep, or swapping cells between timesteps) are
    detected by `detect_collisions()` and marked in red.

    Enhanced to support custom images for Start and Goal markers.
    """

    # Colorblind-safe categorical palette (fixed assignment order).
    ROBOT_PALETTE = [
        '#2a78d6',  # blue
        '#eb6834',  # orange
        '#1baf7a',  # aqua
        '#eda100',  # yellow
        '#e87ba4',  # magenta
        '#008300',  # green
        '#4a3aa7',  # violet
        '#e34948',  # red
    ]
    # Reserved status color for collisions — never assigned to a robot series.
    COLLISION_COLOR = '#d03b3b'
    # Opacity for robots outside their active window (waiting / finished).
    INACTIVE_OPACITY = 0.35

    # Character pack (quantum/images/). Single robot -> Scooby; 2-4 robots ->
    # ninjas, each with its canonical color used for the robot's lane/lines.
    # A robot whose name matches a ninja (case-insensitive) gets that ninja;
    # the rest are assigned in pool order.
    IMAGES_DIR = Path(__file__).parent / 'images'
    SCOOBY_IMAGE = 'scooby.png'
    SNACK_IMAGE = 'scoobysnack.png'    # single-robot goal
    GHOST_IMAGE = 'ghost.png'          # single-robot obstacles
    GARMADON_IMAGE = 'garmadon.png'    # multi-robot (ninja) obstacles
    NINJA_POOL = [
        ('kai', '#e34948'),    # red
        ('jay', '#2a78d6'),    # blue (lightning)
        ('lloyd', '#008300'),  # green
        ('zane', '#1baf7a'),   # ice
        ('cole', '#52514e'),   # earth/black (dark gray so lines stay readable)
    ]
    # Embedded robot images are thumbnailed to this max dimension: animation
    # frames each carry a copy, so full-size PNGs would blow up the HTML.
    ROBOT_IMAGE_MAX_PX = 128

    def __init__(self, grid_size, title="Quantum Pathfinding Visualization", start_image_path=None, goal_image_path=None, obstacle_image_path=None,
                 convention="matrix", robot_images="auto"):
        """
        Initializes the visualizer.
        Args:
            grid_size (tuple): (num_rows, num_cols) of the grid.
            title (str): Title for the plot.
            start_image_path (str, optional): Path to the image file for the Start marker.
            goal_image_path (str, optional): Path to the image file for the Goal marker.
            convention (str): "matrix" (Row/Column, origin top-left) or
                "robotics" (X/Y, Cartesian Y-up, origin bottom-left).
            robot_images: How robots are drawn:
                - "auto" (default): Scooby for a single robot; the ninja pack
                  for 2-4 robots (line colors match each ninja); plain markers
                  for 5+.
                - None: plain markers only.
                - list/dict of image paths: explicit per-robot images (by
                  position for a list, by robot name or index for a dict).
        """
        if convention not in ("matrix", "robotics"):
            raise ValueError(f"convention must be 'matrix' or 'robotics', got {convention!r}")
        self.rows, self.cols = grid_size # Store as rows, cols
        self.title = title
        self.convention = convention
        self.robot_images = robot_images
        self._img_cache = {}
        self.start_image_path = start_image_path
        self.goal_image_path = goal_image_path
        self._start_image_base64 = self._load_image_base64(start_image_path)
        self._goal_image_base64 = self._load_image_base64(goal_image_path)
        self._obstacle_image_base64 = self._load_image_base64(obstacle_image_path)

        # --- Default Styling ---
        self.colors = {
            'background': '#fcfcfb',
            'grid_lines': '#e1e0d9',
            'obstacle': 'black',
            'start': self.ROBOT_PALETTE[0],
            'goal': self.ROBOT_PALETTE[7],
            'path_line': self.ROBOT_PALETTE[1],
            'path_marker': self.ROBOT_PALETTE[1],
            'current_position': self.ROBOT_PALETTE[5],
            'collision': self.COLLISION_COLOR,
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

        # Subway-lane geometry: distance between adjacent robot lanes, and the
        # maximum total spread so lanes never leave the cell.
        self.lane_spacing = 0.12
        self.max_lane_spread = 0.4

        # Multi-robot palette (kept as an instance attribute for backwards
        # compatibility with code that overrides it).
        self.robot_colors = list(self.ROBOT_PALETTE)

    def _load_image_base64(self, image_path, max_px=None):
        """
        Loads an image file and converts it to a base64 string for embedding in Plotly.
        Supports PNG and JPG natively, SVG needs PIL for conversion or special handling.
        Args:
            max_px: If set (and PIL available), thumbnail the image so its
                largest side is max_px — keeps embedded animations small.
        """
        if not image_path:
            return None
        cache_key = (str(image_path), max_px)
        if cache_key in self._img_cache:
            return self._img_cache[cache_key]

        result = self._load_image_base64_uncached(image_path, max_px)
        self._img_cache[cache_key] = result
        return result

    def _load_image_base64_uncached(self, image_path, max_px=None):
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
                    if max_px:
                        img.thumbnail((max_px, max_px))
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
                    elif image_path.lower().endswith(('.jpg', '.jpeg')):
                        mime_type = 'image/jpeg'
                    else:
                        mime_type = 'image/png' # Default guess
                    return f"data:{mime_type};base64,{encoded_data}"
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}. Using fallback marker.")
            return None

    def _convert_coordinates(self, coords):
        """
        Converts matrix coordinates (row, col) to Plotly (x, y) under the
        active display convention. Delegates the robotics/Y-up flip to
        quantum.utils.coordinates.
        """
        if self.convention == "robotics":
            return [to_robotics_xy(c[0], c[1], self.rows) for c in coords]
        # matrix: x = col, y = row (axis is reversed at layout time so row 0
        # renders at the top)
        return [(c[1], c[0]) for c in coords]

    def _axis_config(self):
        """(xaxis, yaxis) layout dicts for the active display convention."""
        xaxis = dict(range=[-0.5, self.cols - 0.5], showgrid=False, zeroline=False,
                     dtick=1, scaleanchor="y", scaleratio=1)
        yaxis = dict(showgrid=False, zeroline=False, dtick=1)
        if self.convention == "robotics":
            xaxis['title'] = "X"
            yaxis.update(title="Y", range=[-0.5, self.rows - 0.5])
        else:
            xaxis['title'] = "Column"
            yaxis.update(title="Row", range=[self.rows - 0.5, -0.5])
        return xaxis, yaxis

    def _cell_px(self):
        """Rendered pixel size of one grid cell — square in every layout."""
        return max(60, min(140, int(600 / max(self.rows, self.cols))))

    def _image_hover_size(self, cell_px=None):
        """
        Marker size (px) matching a cell image's footprint. Layout images
        emit no hover events, so a transparent marker of this size sits under
        each image to make its whole area hoverable.
        Args:
            cell_px: rendered cell size to base it on; defaults to the main
                plot's cell size (step-by-step subplots pass their own).
        """
        cell = cell_px if cell_px is not None else self._cell_px()
        return max(12, int(cell * self.image_marker_size_factor))

    def _effective_image_spec(self, use_images):
        """
        Resolve a per-figure use_images override against the constructor's
        robot_images setting. None -> constructor default; False -> no images
        (pure lines/markers); True -> force images ("auto" if none configured).
        """
        if use_images is False:
            return None
        if use_images is True:
            return self.robot_images or "auto"
        return self.robot_images

    def _layout_dims(self, extra_bottom=0):
        """
        (width, height) so cells stay square regardless of grid shape, with
        room for margins, title and legend.
        """
        cell = self._cell_px()
        width = self.cols * cell + 240
        height = self.rows * cell + 150 + extra_bottom
        return width, height

    # ------------------------------------------------------------------
    # Timeline, subway lanes & collision detection
    # ------------------------------------------------------------------

    @staticmethod
    def _path_to_times(path, start_time=0):
        """
        Maps a path to {t: (row, col)}. The explicit timestep in
        (row, col, t) entries is authoritative; entries without one are
        assigned consecutive timesteps from start_time.
        """
        times = {}
        for i, p in enumerate(path or []):
            t = p[2] if len(p) > 2 else start_time + i
            times[t] = (p[0], p[1])
        return times

    def _robot_offset(self, idx, n_robots):
        """
        Per-robot lane offset (applied to both x and y) so co-located path
        segments render as parallel tracks instead of overlapping.
        A diagonal shift separates lanes on both horizontal and vertical
        corridors.
        """
        if n_robots <= 1:
            return 0.0
        spacing = min(self.lane_spacing, self.max_lane_spread / (n_robots - 1))
        return (idx - (n_robots - 1) / 2.0) * spacing

    @staticmethod
    def _offset_points(points, offset):
        return [(x + offset, y + offset) for x, y in points]

    @staticmethod
    def detect_collisions(robot_paths, start_times=None):
        """
        Detects conflicts between robot paths, mirroring the benchmark
        validator (quantum/benchmark/benchmark.py): strictly time-indexed, so
        a robot only occupies cells at timesteps present in its path. Robots
        with delayed starts or early finishes never produce false conflicts.

        Args:
            robot_paths: dict {robot_key: [(row, col, t) or (row, col), ...]}.
            start_times: optional dict {robot_key: int} used to place paths
                whose entries carry no explicit timestep.

        Returns:
            List of dicts sorted by time:
              {'type': 'vertex', 't': t,   'robots': (a, b), 'cells': [(r, c)]}
              {'type': 'swap',   't': t+1, 'robots': (a, b), 'cells': [(r1, c1), (r2, c2)]}
            'swap' means the robots exchanged cells between t and t+1
            (crossing through each other on the same edge).
        """
        start_times = start_times or {}
        times_by_robot = {
            k: QuantumRoboticsVisualizer._path_to_times(p, start_times.get(k, 0))
            for k, p in robot_paths.items() if p
        }
        return QuantumRoboticsVisualizer._detect_collisions_from_times(times_by_robot)

    @staticmethod
    def _detect_collisions_from_times(times_by_robot):
        keys = list(times_by_robot.keys())
        collisions = []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                ta, tb = times_by_robot[a], times_by_robot[b]
                for t in sorted(set(ta) & set(tb)):
                    if ta[t] == tb[t]:
                        collisions.append({'type': 'vertex', 't': t,
                                           'robots': (a, b), 'cells': [ta[t]]})
                    if (t + 1) in ta and (t + 1) in tb \
                            and ta[t] == tb[t + 1] and tb[t] == ta[t + 1]:
                        collisions.append({'type': 'swap', 't': t + 1,
                                           'robots': (a, b),
                                           'cells': [ta[t], tb[t]]})
        collisions.sort(key=lambda c: c['t'])
        return collisions

    def _collision_trace(self, collisions, max_t=None, showlegend=True):
        """
        Builds a red-X scatter trace for the given collisions.
        For 'swap' conflicts the marker sits on the shared edge midpoint.
        Returns a go.Scatter (empty if no collisions apply).
        """
        xs, ys, texts = [], [], []
        for c in collisions:
            # A swap happens mid-edge, physically at t - 0.5; with fractional
            # (interpolated) max_t the marker appears at that moment. Integer
            # max_t behavior is unchanged.
            t_visible = c['t'] - 0.5 if c['type'] == 'swap' else c['t']
            if max_t is not None and t_visible > max_t:
                continue
            pts = self._convert_coordinates(c['cells'])
            x = sum(p[0] for p in pts) / len(pts)
            y = sum(p[1] for p in pts) / len(pts)
            xs.append(x)
            ys.append(y)
            if c['type'] == 'vertex':
                when = f"t = {c['t']}"
                kind = 'Collision'
            else:
                when = f"t = {c['t'] - 1} → {c['t']}"
                kind = 'Swap collision'
            texts.append(f"⚠ {kind}<br>{c['robots'][0]} × {c['robots'][1]}<br>{when}")
        return go.Scatter(
            x=xs, y=ys, mode='markers',
            marker=dict(color=self.colors['collision'], size=17, symbol='x-thin',
                        line=dict(color=self.colors['collision'], width=4)),
            name='⚠ Collision', showlegend=showlegend and bool(xs),
            text=texts, hovertemplate='%{text}<extra></extra>')

    def _normalize_robots(self, path=None, start=None, goal=None, problem=None, robot_paths=None,
                          use_images=None):
        """
        Normalizes the legacy single-robot arguments and the multi-robot dict
        into {idx: {path, name, color, offset, start, goal, start_conv,
        goal_conv, times ({t: cell}), times_conv ({t: lane-offset (x, y)}),
        t_min, t_max, path_conv}}.
        """
        robot_names = list(problem.robots.keys()) if (problem and hasattr(problem, 'robots')) else []

        robots_data = {}
        if robot_paths:
            for r_idx, (r_key, r_path) in enumerate(robot_paths.items()):
                # get_robot_paths() keys by robot_num; translate to the robot's
                # ID (same robot_num -> name mapping the benchmark uses).
                if isinstance(r_key, int) and r_key < len(robot_names):
                    name = str(robot_names[r_key])
                    problem_idx = r_key
                elif str(r_key) in robot_names:
                    name = str(r_key)
                    problem_idx = robot_names.index(str(r_key))
                else:
                    name = str(r_key)
                    problem_idx = r_idx
                robots_data[r_idx] = {'path': r_path, 'name': name, 'problem_idx': problem_idx}
        elif path:
            robots_data[0] = {'path': path, 'name': str(robot_names[0]) if robot_names else 'Robot',
                              'problem_idx': 0}

        problem_robots_list = list(problem.robots.values()) if (problem and hasattr(problem, 'robots')) else []
        n = len(robots_data)
        self._resolve_robot_images(robots_data, use_images=use_images)

        for idx, data in robots_data.items():
            # An image pack may have pinned the color (ninjas); otherwise palette
            data.setdefault('color', self.robot_colors[idx % len(self.robot_colors)])
            data['offset'] = self._robot_offset(idx, n)

            start_time = 0
            p_idx = data.get('problem_idx', idx)
            if problem and p_idx < len(problem_robots_list):
                robot_obj = problem_robots_list[p_idx]
                data['start'] = robot_obj.start
                data['goal'] = robot_obj.goal
                start_time = getattr(robot_obj, 'start_time', 0) or 0
            else:
                data['start'] = start if idx == 0 else None
                data['goal'] = goal if idx == 0 else None
                if idx == 0 and problem is not None:
                    data['start'] = data['start'] if data['start'] is not None else getattr(problem, "start", None)
                    data['goal'] = data['goal'] if data['goal'] is not None else getattr(problem, "end", None)

            # The first path cell is definitionally the start; no such fallback
            # for goals (an invalid path's last cell need not be the goal).
            if data['start'] is None and data['path']:
                data['start'] = tuple(data['path'][0][:2])

            off = data['offset']
            data['times'] = self._path_to_times(data['path'], start_time)
            ts_sorted = sorted(data['times'])
            data['ts_sorted'] = ts_sorted
            data['t_min'] = ts_sorted[0] if ts_sorted else 0
            data['t_max'] = ts_sorted[-1] if ts_sorted else -1
            data['times_conv'] = {
                t: self._offset_points(self._convert_coordinates([data['times'][t]]), off)[0]
                for t in ts_sorted
            }
            data['path_conv'] = [data['times_conv'][t] for t in ts_sorted]
            data['start_conv'] = self._offset_points(self._convert_coordinates([data['start']]), off)[0] if data.get('start') is not None else None
            data['goal_conv'] = self._offset_points(self._convert_coordinates([data['goal']]), off)[0] if data.get('goal') is not None else None
        return robots_data

    def _resolve_robot_images(self, robots_data, use_images=None):
        """
        Attaches a base64 image (data['image']) — and for the ninja pack a
        matching line color (data['color']) — to each robot according to
        self.robot_images (optionally overridden per figure by use_images).
        Missing/unloadable files fall back to markers.
        """
        spec = self._effective_image_spec(use_images)
        n = len(robots_data)
        if spec is None or n == 0:
            return

        # An explicit goal image applies to the single-robot case in any mode
        if n == 1 and self._goal_image_base64:
            robots_data[0]['goal_image'] = self._goal_image_base64

        if spec == "auto":
            if n == 1:
                # Explicit start_image_path keeps precedence (legacy behavior)
                img = self._start_image_base64 or self._load_image_base64(
                    str(self.IMAGES_DIR / self.SCOOBY_IMAGE), max_px=self.ROBOT_IMAGE_MAX_PX)
                robots_data[0]['image'] = img
                robots_data[0].setdefault('goal_image', self._load_image_base64(
                    str(self.IMAGES_DIR / self.SNACK_IMAGE), max_px=self.ROBOT_IMAGE_MAX_PX))
            elif n < 5:
                pool = dict(self.NINJA_POOL)
                assign = {}
                # Robots named after a ninja get that ninja
                for idx, data in robots_data.items():
                    key = data['name'].lower()
                    if key in pool:
                        assign[idx] = (key, pool.pop(key))
                # The rest take the remaining ninjas in pool order
                remaining = iter(pool.items())
                for idx in robots_data:
                    if idx not in assign:
                        assign[idx] = next(remaining)
                for idx, (ninja, color) in assign.items():
                    img = self._load_image_base64(str(self.IMAGES_DIR / f'{ninja}.png'),
                                                  max_px=self.ROBOT_IMAGE_MAX_PX)
                    robots_data[idx]['image'] = img
                    if img:
                        robots_data[idx]['color'] = color
            return

        # Explicit list (by position) or dict (by robot name or position)
        for idx, data in robots_data.items():
            path = None
            if isinstance(spec, (list, tuple)):
                if idx < len(spec):
                    path = spec[idx]
            elif isinstance(spec, dict):
                path = spec.get(data['name'], spec.get(idx))
            if path:
                data['image'] = self._load_image_base64(str(path), max_px=self.ROBOT_IMAGE_MAX_PX)

    def _resolve_obstacle_image(self, n_robots, use_images=None):
        """
        Effective obstacle image: an explicit obstacle_image_path always wins;
        otherwise the "auto" theme uses the ghost for a single robot and
        Garmadon for 2-4 robots (the ninja scenario). None -> black squares.
        use_images=False suppresses images entirely for the figure.
        """
        if use_images is False:
            return None
        if self._obstacle_image_base64:
            return self._obstacle_image_base64
        if self._effective_image_spec(use_images) == "auto":
            if n_robots == 1:
                return self._load_image_base64(str(self.IMAGES_DIR / self.GHOST_IMAGE),
                                               max_px=self.ROBOT_IMAGE_MAX_PX)
            if 1 < n_robots < 5:
                return self._load_image_base64(str(self.IMAGES_DIR / self.GARMADON_IMAGE),
                                               max_px=self.ROBOT_IMAGE_MAX_PX)
        return None

    @staticmethod
    def _timeline(robots_data):
        """Global [t_min, t_max] across all robots."""
        t0 = min((d['t_min'] for d in robots_data.values() if d['times']), default=0)
        t1 = max((d['t_max'] for d in robots_data.values() if d['times']), default=-1)
        return t0, t1

    def _collisions_for(self, robots_data, enabled):
        if not enabled or len(robots_data) < 2:
            return []
        return self._detect_collisions_from_times(
            {d['name']: d['times'] for d in robots_data.values()})

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
        return self._cell_px()

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

    def _image_dict(self, source, x, y, opacity=1.0, xref="x", yref="y", size_factor=None):
        """Layout-image dict centered on (x, y), sized relative to a grid cell."""
        f = size_factor if size_factor is not None else self.image_marker_size_factor
        return dict(source=source, x=x, y=y, xref=xref, yref=yref,
                    sizex=f, sizey=f, sizing="contain", opacity=opacity,
                    layer="above", xanchor="center", yanchor="middle")

    def _add_image_marker(self, fig, x, y, image_base64_data, name, size_factor=0.8, opacity=1.0):
        """
        Adds an image as a marker to the figure at the specified coordinates.
        The image is centered at (x,y) and sized to fit within a grid cell.
        """
        if image_base64_data:
            fig.add_layout_image(self._image_dict(image_base64_data, x, y,
                                                  opacity=opacity, size_factor=size_factor))

    def _draw_board(self, fig, problem=None, row=None, col=None):
        """Draws terrain background and grid lines (optionally on a subplot)."""
        subplot = dict(row=row, col=col) if row is not None else {}
        if problem is not None and hasattr(problem, 'grid') and problem.grid.terrain is not None:
            for r in range(self.rows):
                for c in range(self.cols):
                    mat_index = problem.grid.get_terrain_at(r, c)
                    if mat_index is not None:
                        color = problem.grid.get_color(mat_index)
                        cx, cy = self._convert_coordinates([(r, c)])[0]
                        fig.add_shape(type="rect", x0=cx-0.5, y0=cy-0.5, x1=cx+0.5, y1=cy+0.5,
                                    fillcolor=color, line=dict(width=0), layer="below", **subplot)

        for c in range(self.cols + 1):
            fig.add_shape(type='line', x0=c-0.5, y0=-0.5, x1=c-0.5, y1=self.rows-0.5,
                        line=dict(color=self.colors['grid_lines'], width=1), **subplot)
        for r in range(self.rows + 1):
            fig.add_shape(type='line', x0=-0.5, y0=r-0.5, x1=self.cols-0.5, y1=r-0.5,
                        line=dict(color=self.colors['grid_lines'], width=1), **subplot)

    def create_static_plot(self, obstacles=None, path=None, start=None, goal=None, current_step=None,
                        problem=None, robot_paths=None, show_collisions=True, use_images=None):
        """
        Creates a single static plot with support for multiple robots.
        Paths are drawn as thin subway-style lanes (per-robot offset) so
        overlapping routes stay readable; real conflicts are marked in red.
        Args:
            obstacles: List of obstacle coordinates
            path: (Legacy) Single robot path.
            start: (Legacy) Single start.
            goal: (Legacy) Single goal.
            current_step: Global timestep to show current positions for. If
                None, the full paths are drawn with no current-position marker.
            problem: Problem instance (needed for multi-robot start/goals)
            robot_paths: Dictionary {robot_id: path_list}
            show_collisions: Mark vertex/swap conflicts with red X markers.
            use_images: Per-figure override of the robot_images setting —
                False for pure lines/markers, True to force images, None to
                follow the constructor.
        """
        robots_data = self._normalize_robots(path, start, goal, problem, robot_paths,
                                             use_images=use_images)
        collisions = self._collisions_for(robots_data, show_collisions)

        fig = go.Figure()
        self._draw_board(fig, problem)

        # --- Obstacles ---
        obstacle_image = self._resolve_obstacle_image(len(robots_data), use_images=use_images)
        obstacles_converted = self._convert_coordinates(obstacles) if obstacles else []
        if obstacles_converted:
            obs_xs, obs_ys = zip(*obstacles_converted)
            hover_text_obs = [f"Obstacle<br>X: {int(x)}<br>Y: {int(y)}" for x, y in zip(obs_xs, obs_ys)]

            if obstacle_image:
                for x, y, text in zip(obs_xs, obs_ys, hover_text_obs):
                    self._add_image_marker(fig, x, y, obstacle_image, 'Obstacle', self.image_marker_size_factor)
                fig.add_trace(go.Scatter(x=obs_xs, y=obs_ys, mode='markers',
                                       marker=dict(color='rgba(0,0,0,0)', size=self._image_hover_size()),
                                       name='Obstacles', showlegend=True, hovertemplate='%{text}<extra></extra>', text=hover_text_obs))
            else:
                fig.add_trace(go.Scatter(x=obs_xs, y=obs_ys, mode='markers',
                                       marker=dict(color=self.colors['obstacle'], size=self._calculate_marker_size('obstacle'), symbol=self.symbols['obstacle']),
                                       name='Obstacles', showlegend=True, hovertemplate='%{text}<extra></extra>', text=hover_text_obs))

        # --- Plot Robots ---
        for idx, data in robots_data.items():
            color = data['color']
            name = data['name']
            start_conv = data['start_conv']
            goal_conv = data['goal_conv']
            ts_sorted = sorted(data['times'])

            # Visible portion of the path and current position (time-indexed)
            if current_step is None:
                visible_ts = ts_sorted
                curr_conv = None
            else:
                visible_ts = [t for t in ts_sorted if t <= current_step]
                curr_conv = data['times_conv'][visible_ts[-1]] if visible_ts else None

            # 1. Path (subway lane: thin line with per-step hover)
            if len(visible_ts) > 1:
                pts = [data['times_conv'][t] for t in visible_ts]
                pxs, pys = zip(*pts)
                fig.add_trace(go.Scatter(x=pxs, y=pys, mode='lines',
                                       line=dict(color=color, width=2.5, shape='linear'),
                                       name=name, showlegend=True,
                                       hovertemplate=f'{name}<extra></extra>'))
                steps_txt = [f"{name}<br>t = {t}" for t in visible_ts]
                fig.add_trace(go.Scatter(x=pxs, y=pys, mode='markers',
                                       marker=dict(color=color, size=5, symbol='circle'),
                                       showlegend=False, text=steps_txt,
                                       hovertemplate='%{text}<extra></extra>'))

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
                goal_image = data.get('goal_image')
                if goal_image:
                    self._add_image_marker(fig, goal_conv[0], goal_conv[1], goal_image, f'{name} Goal',
                                           self.image_marker_size_factor, opacity=opacity)
                    fig.add_trace(go.Scatter(x=[goal_conv[0]], y=[goal_conv[1]], mode='markers',
                                           marker=dict(color='rgba(0,0,0,0)', size=self._image_hover_size()),
                                           showlegend=False, hovertemplate=f'{name} Goal<extra></extra>'))
                else:
                    fig.add_trace(go.Scatter(x=[goal_conv[0]], y=[goal_conv[1]], mode='markers',
                                           marker=dict(color=color, size=self._calculate_marker_size('goal'), symbol='diamond', opacity=opacity),
                                           name=f'{name} Goal', showlegend=False, hovertemplate=f'{name} Goal<extra></extra>'))

            # 4. Current Position
            if curr_conv:
                image = data.get('image') or (self._start_image_base64 if idx == 0 else None)
                if image:
                    self._add_image_marker(fig, curr_conv[0], curr_conv[1], image, name, self.image_marker_size_factor)
                    # Invisible marker sized to the image keeps its whole area hoverable
                    fig.add_trace(go.Scatter(x=[curr_conv[0]], y=[curr_conv[1]], mode='markers',
                                           marker=dict(color='rgba(0,0,0,0)', size=self._image_hover_size()),
                                           showlegend=False, hovertemplate=f'{name} (Current)<extra></extra>'))
                else:
                    fig.add_trace(go.Scatter(x=[curr_conv[0]], y=[curr_conv[1]], mode='markers',
                                           marker=dict(color=color, size=self._calculate_marker_size('current_position'), symbol='star',
                                                       line=dict(color='white', width=1)),
                                           name=f'{name} (current)', showlegend=False, hovertemplate=f'{name} (Current)<extra></extra>'))

        # --- Collisions ---
        if collisions:
            fig.add_trace(self._collision_trace(collisions, max_t=current_step))

        # --- Final Layout ---
        xaxis, yaxis = self._axis_config()
        width, height = self._layout_dims()
        fig.update_layout(
            title=self.title,
            xaxis=xaxis, yaxis=yaxis,
            showlegend=True,
            plot_bgcolor=self.colors['background'],
            margin=dict(l=50, r=50, t=80, b=50),
            width=width, height=height,
            hovermode='closest'
        )
        return fig

    def create_animated_plot(self, obstacles=None, path=None, start=None, goal=None,
                             problem=None, robot_paths=None, frame_duration=500,
                             show_collisions=True, smooth=True, substeps=4,
                             use_images=None):
        """
        Creates an animated timeline of the paths: play/pause buttons plus a
        time slider. Each robot is a moving marker with a growing trail; the
        full route is shown as a faint background lane. Robots outside their
        active window (not yet started / already finished) are shown faded.
        Conflicts flash red at their timestep and stay marked afterwards.

        Note: custom robot images are not animated (Plotly layout images do not
        participate in frames); markers are used for the moving robots.

        Args:
            frame_duration: milliseconds per timestep when playing.
            smooth: True (default) interpolates robot motion between cells for
                continuous movement. False shows the raw discrete timeline —
                one frame per QUBO timestep — better for analyzing the
                formulation itself.
            substeps: interpolation frames per timestep when smooth=True.
            use_images: Per-figure override of the robot_images setting —
                False for pure lines/markers, True to force images, None to
                follow the constructor.
            (remaining args as in create_static_plot)
        Returns:
            go.Figure with frames, slider and play controls. The slider always
            steps on whole timesteps, even when smooth.
        """
        robots_data = self._normalize_robots(path, start, goal, problem, robot_paths,
                                             use_images=use_images)
        if not robots_data or all(not d['times'] for d in robots_data.values()):
            raise ValueError("A path (or robot_paths) is required for the animated plot.")

        t0, t1 = self._timeline(robots_data)
        collisions = self._collisions_for(robots_data, show_collisions)

        fig = go.Figure()
        self._draw_board(fig, problem)

        # --- Static base traces ---
        obstacle_imgs = []
        obstacle_image = self._resolve_obstacle_image(len(robots_data), use_images=use_images)
        obstacles_converted = self._convert_coordinates(obstacles) if obstacles else []
        if obstacles_converted:
            obs_xs, obs_ys = zip(*obstacles_converted)
            if obstacle_image:
                # Collected as dicts: animation frames replace layout.images
                # wholesale, so obstacles must be re-included in every frame.
                for ox, oy in zip(obs_xs, obs_ys):
                    obstacle_imgs.append(self._image_dict(obstacle_image, ox, oy))
                fig.add_trace(go.Scatter(x=obs_xs, y=obs_ys, mode='markers',
                                       marker=dict(color='rgba(0,0,0,0)', size=self._image_hover_size()),
                                       name='Obstacles', showlegend=False,
                                       hovertemplate='Obstacle<extra></extra>'))
            else:
                fig.add_trace(go.Scatter(x=obs_xs, y=obs_ys, mode='markers',
                                       marker=dict(color=self.colors['obstacle'], size=self._calculate_marker_size('obstacle'), symbol='square'),
                                       name='Obstacles', hovertemplate='Obstacle<extra></extra>'))

        for data in robots_data.values():
            color, name = data['color'], data['name']
            # Faint full route (context while animating)
            if len(data['path_conv']) > 1:
                xs, ys = zip(*data['path_conv'])
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
                                       line=dict(color=color, width=2), opacity=0.25,
                                       showlegend=False,
                                       hovertemplate=f'{name} (route)<extra></extra>'))
            if data['start_conv']:
                fig.add_trace(go.Scatter(x=[data['start_conv'][0]], y=[data['start_conv'][1]],
                                       mode='markers+text', text=['S'], textfont=dict(color=color, size=10),
                                       marker=dict(color=color, size=12, symbol='circle-open', line=dict(width=2)),
                                       showlegend=False, hovertemplate=f'{name} Start<extra></extra>'))
            if data['goal_conv']:
                if data.get('goal_image'):
                    # Static image, but frames replace layout.images wholesale,
                    # so it joins the per-frame list next to the obstacles.
                    obstacle_imgs.append(self._image_dict(data['goal_image'],
                                                          data['goal_conv'][0], data['goal_conv'][1]))
                    fig.add_trace(go.Scatter(x=[data['goal_conv'][0]], y=[data['goal_conv'][1]],
                                           mode='markers',
                                           marker=dict(color='rgba(0,0,0,0)', size=self._image_hover_size()),
                                           showlegend=False, hovertemplate=f'{name} Goal<extra></extra>'))
                else:
                    fig.add_trace(go.Scatter(x=[data['goal_conv'][0]], y=[data['goal_conv'][1]],
                                           mode='markers',
                                           marker=dict(color=color, size=11, symbol='diamond'),
                                           showlegend=False, hovertemplate=f'{name} Goal<extra></extra>'))

        # --- Animated traces (trail + head per robot, then collisions) ---
        import bisect

        def robot_state(data, tau):
            """(trail_points, head_xy, active) at possibly-fractional time tau."""
            ts = data['ts_sorted']
            if not ts:
                return [], None, False
            conv = data['times_conv']
            if tau < ts[0]:
                # Waiting to start: sit faded at the first cell, no trail
                return [], conv[ts[0]], False
            if tau >= ts[-1]:
                return [conv[tt] for tt in ts], conv[ts[-1]], tau <= ts[-1]
            i = bisect.bisect_right(ts, tau)
            ta, tb = ts[i - 1], ts[i]
            pa, pb = conv[ta], conv[tb]
            frac = (tau - ta) / (tb - ta)
            head = (pa[0] + frac * (pb[0] - pa[0]), pa[1] + frac * (pb[1] - pa[1]))
            trail = [conv[tt] for tt in ts[:i]]
            if frac > 0:
                trail = trail + [head]
            return trail, head, True

        def frame_traces(tau):
            traces = []
            for data in robots_data.values():
                color, name = data['color'], data['name']
                trail, head, active = robot_state(data, tau)
                traces.append(go.Scatter(
                    x=[p[0] for p in trail], y=[p[1] for p in trail], mode='lines',
                    line=dict(color=color, width=3),
                    name=name, showlegend=True,
                    hovertemplate=f'{name}<extra></extra>'))

                if head is not None:
                    if active:
                        status = f't = {tau:g}'
                    elif tau < data['t_min']:
                        status = f'starts at t = {data["t_min"]}'
                    else:
                        status = f'finished at t = {data["t_max"]}'
                    cell = data['times'].get(int(tau))
                    cell_txt = f'<br>cell = ({cell[0]}, {cell[1]})' if cell else ''
                    if data.get('image'):
                        # Sprite drawn as a layout image; a transparent marker
                        # its size makes the whole character hoverable
                        head_marker = dict(color='rgba(0,0,0,0)', size=self._image_hover_size())
                    else:
                        head_marker = dict(color=color, size=14, symbol='circle',
                                           line=dict(color='white', width=2))
                    traces.append(go.Scatter(
                        x=[head[0]], y=[head[1]], mode='markers',
                        marker=head_marker,
                        opacity=1.0 if active else self.INACTIVE_OPACITY,
                        showlegend=False,
                        hovertemplate=f'{name}<br>{status}{cell_txt}<extra></extra>'))
                else:
                    traces.append(go.Scatter(x=[], y=[], mode='markers', showlegend=False))

            traces.append(self._collision_trace(collisions, max_t=tau))
            return traces

        # Frame timeline: whole timesteps, or interpolated quarters etc.
        if smooth and substeps > 1 and t1 > t0:
            taus = [t0 + k / substeps for k in range((t1 - t0) * substeps + 1)]
            per_frame = max(20, frame_duration // substeps)
            transition = 0  # the sub-frames themselves are the interpolation
        else:
            taus = [float(t) for t in range(t0, t1 + 1)]
            per_frame = frame_duration
            transition = min(200, frame_duration // 2)

        first = frame_traces(taus[0])
        anim_start = len(fig.data)
        for tr in first:
            fig.add_trace(tr)
        anim_indices = list(range(anim_start, len(fig.data)))

        # Robot character images ride along in each frame's layout.images
        # (traces can't animate images); obstacles are re-included since the
        # list is replaced per frame.
        any_robot_imgs = any(d.get('image') for d in robots_data.values())

        def frame_images(tau):
            imgs = list(obstacle_imgs)
            for data in robots_data.values():
                if data.get('image'):
                    _, head, active = robot_state(data, tau)
                    if head is not None:
                        imgs.append(self._image_dict(data['image'], head[0], head[1],
                                                     opacity=1.0 if active else self.INACTIVE_OPACITY))
            return imgs

        for img in frame_images(taus[0]):
            fig.add_layout_image(img)

        fig.frames = [go.Frame(name=f'{tau:g}', data=frame_traces(tau), traces=anim_indices,
                               layout=dict(images=frame_images(tau)) if any_robot_imgs else None)
                      for tau in taus]

        # --- Controls (slider steps on whole timesteps only) ---
        # Layout images only refresh on redraw, so it must be on when robots
        # are drawn as characters.
        redraw = any_robot_imgs
        frame_args = {'frame': {'duration': per_frame, 'redraw': redraw},
                      'mode': 'immediate', 'fromcurrent': True,
                      'transition': {'duration': transition, 'easing': 'linear'}}
        fig.update_layout(
            updatemenus=[dict(
                type='buttons', direction='left',
                x=0.0, y=-0.12, xanchor='left', yanchor='top', pad=dict(r=10, t=10),
                buttons=[
                    dict(label='▶ Play', method='animate', args=[None, frame_args]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], {'frame': {'duration': 0, 'redraw': redraw}, 'mode': 'immediate'}]),
                ])],
            sliders=[dict(
                x=0.22, y=-0.1, xanchor='left', yanchor='top', len=0.78,
                currentvalue=dict(prefix='t = ', font=dict(size=14)),
                steps=[dict(label=f'{tau:g}', method='animate',
                            args=[[f'{tau:g}'], {'frame': {'duration': 0, 'redraw': redraw}, 'mode': 'immediate'}])
                       for tau in taus if tau.is_integer()])],
        )

        xaxis, yaxis = self._axis_config()
        width, height = self._layout_dims(extra_bottom=60)
        fig.update_layout(
            title=self.title,
            xaxis=xaxis, yaxis=yaxis,
            showlegend=True,
            plot_bgcolor=self.colors['background'],
            margin=dict(l=50, r=50, t=80, b=110),
            width=width, height=height,
            hovermode='closest'
        )
        return fig

    def create_step_by_step_plot(self, obstacles, path=None, start=None, goal=None, problem=None, robot_paths=None,
                                 show_collisions=True, use_images=None):
        """
        Creates a subplot visualization showing the path evolution over time.
        One subplot per global timestep; uses the same subway-style lane
        offsets as the static plot, and marks conflicts occurring at each
        timestep with red X markers.
        use_images: per-figure override of robot_images (False -> lines only).
        """
        robots_data = self._normalize_robots(path, start, goal, problem, robot_paths,
                                             use_images=use_images)
        if not robots_data or all(not d['times'] for d in robots_data.values()):
             raise ValueError("Path is required for step-by-step visualization.")

        t0, t1 = self._timeline(robots_data)
        timeline = list(range(t0, t1 + 1))
        max_steps = len(timeline)
        collisions = self._collisions_for(robots_data, show_collisions)

        import math
        cols_subplot = math.ceil(math.sqrt(max_steps))
        rows_subplot = math.ceil(max_steps / cols_subplot)

        # Calculate consistent sizing based on grid size
        base_width, base_height = self._calculate_figure_size()

        fig = make_subplots(
            rows=rows_subplot, cols=cols_subplot,
            subplot_titles=[f"t = {t}" for t in timeline],
            horizontal_spacing=0.02,
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False} for _ in range(cols_subplot)] for _ in range(rows_subplot)]
        )

        # Convert obstacles once
        obstacle_image = self._resolve_obstacle_image(len(robots_data), use_images=use_images)
        obstacles_converted = self._convert_coordinates(obstacles) if obstacles else []
        # Subplot cells are smaller than the main plot's; size hover zones to them
        step_hover = self._image_hover_size(min(base_width / self.cols, base_height / self.rows))

        # Loop through each timestep
        for i, t in enumerate(timeline):
            row_subplot = (i // cols_subplot) + 1
            col_subplot = (i % cols_subplot) + 1

            subplot_index = (row_subplot - 1) * cols_subplot + (col_subplot - 1) + 1
            xref = f"x{subplot_index}" if subplot_index > 1 else "x"
            yref = f"y{subplot_index}" if subplot_index > 1 else "y"

            # 1. Terrain + Grid Lines
            self._draw_board(fig, problem, row=row_subplot, col=col_subplot)

            # 3. Obstacles
            if obstacles_converted:
                obs_xs, obs_ys = zip(*obstacles_converted)
                if obstacle_image:
                    for ox, oy in zip(obs_xs, obs_ys):
                        fig.add_layout_image(self._image_dict(obstacle_image, ox, oy, xref=xref, yref=yref))
                    fig.add_trace(go.Scatter(x=list(obs_xs), y=list(obs_ys), mode='markers',
                                           marker=dict(color='rgba(0,0,0,0)', size=step_hover),
                                           showlegend=(i==0), name='Obstacles',
                                           hovertemplate='Obstacle<extra></extra>'), row=row_subplot, col=col_subplot)
                else:
                    fig.add_trace(go.Scatter(x=list(obs_xs), y=list(obs_ys), mode='markers',
                                           marker=dict(color=self.colors['obstacle'], size=self._calculate_marker_size('obstacle'), symbol='square'),
                                           showlegend=(i==0), name='Obstacles',
                                           hovertemplate='Obstacle<extra></extra>'), row=row_subplot, col=col_subplot)

            # 4. Robots
            for idx, data in robots_data.items():
                color = data['color']
                name = data['name']
                start_conv = data['start_conv']
                goal_conv = data['goal_conv']
                ts_sorted = sorted(data['times'])
                trail_ts = [tt for tt in ts_sorted if tt <= t]

                # Start
                if start_conv and i == 0:
                     fig.add_trace(go.Scatter(x=[start_conv[0]], y=[start_conv[1]], mode='markers',
                                            marker=dict(color=color, size=6, symbol='circle-open', line=dict(width=2)),
                                            showlegend=False,
                                            hovertemplate=f'{name} Start<extra></extra>'), row=row_subplot, col=col_subplot)

                # Goal
                if goal_conv:
                    opacity = 1.0
                    # Check if reached at this step
                    curr_conv = data['times_conv'][trail_ts[-1]] if trail_ts else None
                    if curr_conv == goal_conv:
                        opacity = 0.3

                    if data.get('goal_image'):
                        fig.add_layout_image(self._image_dict(data['goal_image'], goal_conv[0], goal_conv[1],
                                                              opacity=opacity, xref=xref, yref=yref))
                        fig.add_trace(go.Scatter(x=[goal_conv[0]], y=[goal_conv[1]], mode='markers',
                                               marker=dict(color='rgba(0,0,0,0)', size=step_hover),
                                               showlegend=False,
                                               hovertemplate=f'{name} Goal<extra></extra>'), row=row_subplot, col=col_subplot)
                    else:
                        fig.add_trace(go.Scatter(x=[goal_conv[0]], y=[goal_conv[1]], mode='markers',
                                               marker=dict(color=color, size=8, symbol='diamond', opacity=opacity),
                                               showlegend=False,
                                               hovertemplate=f'{name} Goal<extra></extra>'), row=row_subplot, col=col_subplot)

                # Path history up to time t
                if len(trail_ts) > 1:
                    pts = [data['times_conv'][tt] for tt in trail_ts]
                    pxs, pys = zip(*pts)
                    fig.add_trace(go.Scatter(x=pxs, y=pys, mode='lines', line=dict(color=color, width=2),
                                           showlegend=(i==0), name=name,
                                           hovertemplate=f'{name}<extra></extra>'), row=row_subplot, col=col_subplot)

                # Current Position (faded once the robot has finished)
                if trail_ts:
                    curr = data['times_conv'][trail_ts[-1]]
                    active = t <= data['t_max']
                    curr_hover = f'{name} (t = {trail_ts[-1]})<extra></extra>'
                    image = data.get('image') or (self._start_image_base64 if idx == 0 else None)
                    if image:
                        fig.add_layout_image(self._image_dict(image, curr[0], curr[1],
                                                              opacity=1.0 if active else self.INACTIVE_OPACITY,
                                                              xref=xref, yref=yref))
                        fig.add_trace(go.Scatter(x=[curr[0]], y=[curr[1]], mode='markers',
                                               marker=dict(color='rgba(0,0,0,0)', size=step_hover),
                                               showlegend=False,
                                               hovertemplate=curr_hover), row=row_subplot, col=col_subplot)
                    else:
                        fig.add_trace(go.Scatter(x=[curr[0]], y=[curr[1]], mode='markers',
                                               marker=dict(color=color, size=10, symbol='star'),
                                               opacity=1.0 if active else self.INACTIVE_OPACITY,
                                               showlegend=False,
                                               hovertemplate=curr_hover), row=row_subplot, col=col_subplot)

            # 5. Collisions occurring at this timestep
            step_collisions = [c for c in collisions if c['t'] == t]
            if step_collisions:
                fig.add_trace(self._collision_trace(step_collisions, showlegend=False),
                              row=row_subplot, col=col_subplot)

            # Axis updates
            y_range = [-0.5, self.rows - 0.5] if self.convention == "robotics" else [self.rows - 0.5, -0.5]
            fig.update_xaxes(range=[-0.5, self.cols-0.5], showgrid=False, zeroline=False, dtick=1, row=row_subplot, col=col_subplot)
            fig.update_yaxes(range=y_range, showgrid=False, zeroline=False, dtick=1, row=row_subplot, col=col_subplot)

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

    def write_gif(self, fig, filename, timestep_duration=600, scale=1):
        """
        Exports an animated figure (from create_animated_plot) as a GIF —
        the shareable counterpart to the interactive HTML (README, slides,
        paper supplementary). Renders each frame to PNG via kaleido and
        stitches them with PIL.

        Args:
            fig: Figure with frames (create_animated_plot output, smooth or
                discrete — sub-frame timing is inferred automatically).
            filename: Output .gif path.
            timestep_duration: milliseconds per whole timestep.
            scale: kaleido render scale (2 = double resolution).
        """
        if not PIL_AVAILABLE:
            print("Failed to save GIF: PIL (Pillow) is required. Install with `pip install pillow`.")
            return
        if not fig.frames:
            print("Failed to save GIF: figure has no animation frames. Use create_animated_plot().")
            return

        import copy
        base = fig.to_dict()
        layout = copy.deepcopy(base['layout'])
        # Interactive controls make no sense in a GIF; a t-label replaces the slider readout
        layout.pop('updatemenus', None)
        layout.pop('sliders', None)

        # Sub-frames per timestep: total frames vs whole-timestep slider steps
        n_frames = len(base['frames'])
        sliders = fig.layout.sliders
        n_steps = len(sliders[0].steps) if sliders else n_frames
        sub = max(1, round((n_frames - 1) / max(1, n_steps - 1)))
        per_frame = max(20, timestep_duration // sub)

        images = []
        try:
            for fr in base['frames']:
                data = copy.deepcopy(base['data'])
                idxs = fr.get('traces') or list(range(len(fr['data'])))
                for i, tr in zip(idxs, fr['data']):
                    data[i] = tr
                frame_fig = go.Figure(data=data, layout=layout)
                if fr.get('layout'):
                    # Per-frame layout carries the moving robot images
                    frame_fig.update_layout(fr['layout'])
                t_label = int(float(fr.get('name', 0)))
                frame_fig.add_annotation(text=f"t = {t_label}",
                                         xref='paper', yref='paper', x=1, y=1.06,
                                         xanchor='right', yanchor='bottom',
                                         showarrow=False, font=dict(size=16))
                png_bytes = frame_fig.to_image(format='png', scale=scale)
                images.append(Image.open(BytesIO(png_bytes)).convert('RGB'))
        except Exception as e:
            print(f"Failed to render GIF frames: {e}. Ensure kaleido (`pip install kaleido`) is installed.")
            return

        images[0].save(filename, save_all=True, append_images=images[1:],
                       duration=per_frame, loop=0)
        print(f"GIF saved to {filename} ({len(images)} frames)")

    def write_image(self, fig, filename, format='png', width=None, height=None):
        """Saves the figure as a static image."""
        try:
            fig.write_image(filename, format=format, width=width, height=height)
            print(f"Image saved to {filename}")
        except Exception as e:
             print(f"Failed to save image: {e}. Ensure kaleido (`pip install kaleido`) is installed.")
