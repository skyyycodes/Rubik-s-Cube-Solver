"""
3D Rubik's Cube Input Interface
An intuitive way to input your cube state using a 3D visualization.

Controls:
- Click on a sticker to select it
- Click on a color in the palette to set the selected sticker's color
- Left/Right drag to rotate the cube view
- Tab / Shift+Tab: Navigate between faces
- 1-6: Quick color input (1:White 2:Red 3:Blue 4:Yellow 5:Orange 6:Green)
- Arrow keys: Navigate within a face
- Enter: Solve the cube
- R: Reset to solved state
- G: Toggle guided mode (step-by-step face input)
- Q: Quit
"""

import threading
import time

import numpy as np
import matplotlib

# Force an interactive backend for the standalone GUI. Frozen builds can
# otherwise fall back to non-interactive Agg and exit without showing a window.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

import sys as _sys
from pathlib import Path as _Path
# Ensure the local enhanced kociemba is found before the pip version
_sys.path.insert(0, str(_Path(__file__).resolve().parent / 'kociemba'))
from kociemba import solve as kociemba_solve
from kociemba.pykociemba import tools as _ktools


def solve_cube_compat(cube_string, mode='k5'):
    """Solve with the given mode; fall back to basic API on older kociemba.

    Modes:
        'baseline' — K=1, no extras (original Kociemba)
        'k5'       — K=5 multi-solution search
        'anytime'  — K=5 + 2 s time budget
        'neural'   — K=5 + LUT-based neural move ordering
    """
    try:
        if mode == 'baseline':
            return kociemba_solve(cube_string, max_phase1_solutions=1)
        elif mode == 'k5':
            return kociemba_solve(cube_string, max_phase1_solutions=5,
                                 timeout=15000)
        elif mode == 'anytime':
            return kociemba_solve(cube_string, max_phase1_solutions=5,
                                 time_budget_sec=2.0)
        elif mode == 'neural':
            return kociemba_solve(cube_string, max_phase1_solutions=5,
                                 timeout=15000, neural_strategy='lut')
        else:
            return kociemba_solve(cube_string, max_phase1_solutions=5,
                                 timeout=15000)
    except TypeError:
        return kociemba_solve(cube_string)

try:
    # Local source layout in this repository
    from kociemba.kociemba.pykociemba.facecube import FaceCube
    from kociemba.kociemba.pykociemba.cubiecube import moveCube
except ImportError:
    # pip-installed package layout
    from kociemba.pykociemba.facecube import FaceCube
    from kociemba.pykociemba.cubiecube import moveCube


class RubiksCube3D:
    def __init__(self):
        # Color mapping: face -> RGB color
        self.color_map = {
            'white': '#FFFFFF',
            'yellow': '#FFD500',
            'blue': '#0046AD',
            'green': '#009B48',
            'red': '#B71234',
            'orange': '#FF5800'
        }

        # Darker versions for 3D depth effect
        self.color_map_dark = {
            'white': '#E8E8E8',
            'yellow': '#E8C200',
            'blue': '#003080',
            'green': '#007830',
            'red': '#900020',
            'orange': '#D04000'
        }

        # Initialize cube state - each face is 3x3 grid
        self.faces = {
            'U': [['white']*3 for _ in range(3)],   # Up - white center
            'D': [['yellow']*3 for _ in range(3)],  # Down - yellow center
            'F': [['blue']*3 for _ in range(3)],    # Front - blue center
            'B': [['green']*3 for _ in range(3)],   # Back - green center
            'R': [['red']*3 for _ in range(3)],     # Right - red center
            'L': [['orange']*3 for _ in range(3)]   # Left - orange center
        }

        # Face display names (visual orientation: Blue top, Red front, Yellow right)
        self.face_names = {
            'U': 'LEFT (White)',
            'D': 'RIGHT (Yellow)',
            'F': 'TOP (Blue)',
            'B': 'BOTTOM (Green)',
            'R': 'FRONT (Red)',
            'L': 'BACK (Orange)'
        }

        # Current selection
        self.current_face = 'F'
        self.current_row = 0
        self.current_col = 0

        # Selected color for painting
        self.selected_color = 'white'

        # Color cycle for input
        self.colors = ['white', 'red', 'blue', 'yellow', 'orange', 'green']

        # View angles - physical cube: blue top, red front, yellow right
        self.elev = 25
        self.azim = 45

        # Dragging state
        self.dragging = False
        self.last_mouse_pos = None

        # Guided mode
        self.guided_mode = False
        self.guided_face_index = 0
        self.guided_position = 0
        self.face_order = ['U', 'F', 'R', 'B', 'L', 'D']

        self.fig = None
        self.ax_3d = None
        self.ax_2d = None
        self.ax_palette = None
        self.ax_info = None

        # Solver mode toggle (M key cycles through these)
        self.solve_modes = ['baseline', 'k5', 'anytime', 'neural']
        self.solve_mode_index = 1  # default: k5
        self.solve_mode_labels = {
            'baseline': 'Baseline (K=1)',
            'k5':       'Multi-Solution (K=5)',
            'anytime':  'Anytime (K=5, 2s)',
            'neural':   'Neural LUT (K=5)',
        }

        # Message to display
        self.message = ""
        self.message_color = 'black'

        # Solution playback state
        self.solution_moves = []
        self.current_move_index = -1  # -1 = not started
        self.base_cube_string = None
        self.playback_cube_string = None
        self.last_solve_info = None

        # Async solve / progress-bar state
        self.solve_in_progress = False
        self.solve_result = None        # ('ok', solution, elapsed_ms) or ('err', msg, 0)
        self.solve_thread = None
        self.progress_timer = None
        self.progress_position = 0
        self.solve_start_time = 0.0
        self._pending_cube_string = None
        self.ax_progress = None
        self.progress_track = None
        self.progress_marquee = None
        self.progress_text = None
        self._progress_bg = None
        self._original_switch_interval = None

    def _physical_faces(self):
        """
        Mapping between our internal face keys and Kociemba letters for THIS UI.

        Order is the physical face order that Kociemba expects:
        Top, Right, Front, Bottom, Left, Back  =>  U, R, F, D, L, B
        """
        return [
            ('F', 'U'),  # Top    -> our F  (Blue)
            ('D', 'R'),  # Right  -> our D  (Yellow)
            ('R', 'F'),  # Front  -> our R  (Red)
            ('B', 'D'),  # Bottom -> our B  (Green)
            ('U', 'L'),  # Left   -> our U  (White)
            ('L', 'B'),  # Back   -> our L  (Orange)
        ]

    def _apply_kociemba_move_to_string(self, cube_string: str, move: str) -> str:
        """Apply a single Kociemba move (e.g. U, R', F2) to a cube string."""
        axis_map = {'U': 0, 'R': 1, 'F': 2, 'D': 3, 'L': 4, 'B': 5}

        if not move:
            return cube_string

        face = move[0]
        suffix = move[1:] if len(move) > 1 else ''
        if face not in axis_map:
            return cube_string

        turns = 1
        if suffix == "2":
            turns = 2
        elif suffix == "'":
            turns = 3

        cc = FaceCube(cube_string).toCubieCube()
        for _ in range(turns):
            cc.multiply(moveCube[axis_map[face]])
        return cc.toFaceCube().to_String()

    def _update_faces_from_cube_string(self, cube_string: str):
        """Update self.faces (our UI faces) from a Kociemba cube string."""
        if not cube_string or len(cube_string) != 54:
            return

        # Ensure we have the latest letter→color mapping from centers.
        # (to_kociemba_string sets self.letter_to_color)
        if not hasattr(self, 'letter_to_color') or not self.letter_to_color:
            _ = self.to_kociemba_string()

        # Split cube string into faces in Kociemba order: U, R, F, D, L, B
        chunks = {
            'U': cube_string[0:9],
            'R': cube_string[9:18],
            'F': cube_string[18:27],
            'D': cube_string[27:36],
            'L': cube_string[36:45],
            'B': cube_string[45:54],
        }

        # Place them back into our internal faces using the physical mapping.
        for face_name, letter in self._physical_faces():
            s = chunks[letter]
            colors = [self.letter_to_color.get(ch, 'white') for ch in s]
            self.faces[face_name] = [
                [colors[0], colors[1], colors[2]],
                [colors[3], colors[4], colors[5]],
                [colors[6], colors[7], colors[8]],
            ]

    def get_face_sticker_coords(self, face_name):
        """Get 3D coordinates for each sticker on a face for hit testing"""
        coords = []

        # Physical cube orientation: Blue(F)=Top, Green(B)=Bottom, Red(R)=Front,
        #                            Orange(L)=Back, Yellow(D)=visual Right, White(U)=visual Left
        if face_name == 'F':  # Blue - TOP at z=1
            for i in range(3):
                for j in range(3):
                    center = np.array([(2-j)/3 + 1/6, (2-i)/3 + 1/6, 1])
                    coords.append((i, j, center))
        elif face_name == 'B':  # Green - BOTTOM at z=0
            for i in range(3):
                for j in range(3):
                    center = np.array([(2-j)/3 + 1/6, i/3 + 1/6, 0])
                    coords.append((i, j, center))
        elif face_name == 'R':  # Red - FRONT at y=1
            for i in range(3):
                for j in range(3):
                    center = np.array([(2-j)/3 + 1/6, 1, i/3 + 1/6])
                    coords.append((i, j, center))
        elif face_name == 'L':  # Orange - BACK at y=0
            for i in range(3):
                for j in range(3):
                    center = np.array([j/3 + 1/6, 0, i/3 + 1/6])
                    coords.append((i, j, center))
        elif face_name == 'U':  # White - at x=1 (visual left)
            for i in range(3):
                for j in range(3):
                    center = np.array([1, j/3 + 1/6, i/3 + 1/6])
                    coords.append((i, j, center))
        elif face_name == 'D':  # Yellow - at x=0 (visual right)
            for i in range(3):
                for j in range(3):
                    center = np.array([0, (2-j)/3 + 1/6, i/3 + 1/6])
                    coords.append((i, j, center))

        return coords

    def draw_face_3d(self, ax, face_name, origin, dir1, dir2):
        """Draw a 3x3 face of the cube in 3D"""
        face = self.faces[face_name]

        for i in range(3):
            for j in range(3):
                # Calculate corners of this cell
                p0 = origin + (j/3)*dir1 + ((2-i)/3)*dir2
                p1 = origin + ((j+1)/3)*dir1 + ((2-i)/3)*dir2
                p2 = origin + ((j+1)/3)*dir1 + ((2-i+1)/3)*dir2
                p3 = origin + (j/3)*dir1 + ((2-i+1)/3)*dir2

                verts = [[p0, p1, p2, p3]]
                color = self.color_map[face[i][j]]

                # Highlight selected cell
                edge_color = '#303030'
                line_width = 0.8
                if face_name == self.current_face and i == self.current_row and j == self.current_col:
                    edge_color = '#00FFFF'
                    line_width = 3

                poly = Poly3DCollection(verts, facecolors=color,
                                        edgecolors=edge_color, linewidths=line_width)
                ax.add_collection3d(poly)

    def draw_cube_3d(self):
        """Draw the entire 3D cube - Physical cube orientation: Blue=Top, Red=Front"""
        self.ax_3d.clear()

        # Draw black cube body
        self.draw_cube_body()

        # Coordinate system: z=up, y=front
        # Physical orientation: Blue(F)=Top, Green(B)=Bottom, Red(R)=Front,
        #                       Orange(L)=Back, Yellow(D)=visual Right, White(U)=visual Left

        # F (blue) - TOP at z=1
        self.draw_face_3d(self.ax_3d, 'F',
                         np.array([1, 1, 1]),
                         np.array([-1, 0, 0]),
                         np.array([0, -1, 0]))

        # B (green) - BOTTOM at z=0
        self.draw_face_3d(self.ax_3d, 'B',
                         np.array([1, 0, 0]),
                         np.array([-1, 0, 0]),
                         np.array([0, 1, 0]))

        # R (red) - FRONT at y=1
        self.draw_face_3d(self.ax_3d, 'R',
                         np.array([1, 1, 0]),
                         np.array([-1, 0, 0]),
                         np.array([0, 0, 1]))

        # L (orange) - BACK at y=0
        self.draw_face_3d(self.ax_3d, 'L',
                         np.array([0, 0, 0]),
                         np.array([1, 0, 0]),
                         np.array([0, 0, 1]))

        # U (white) - at x=1 (appears on visual left due to view angle)
        self.draw_face_3d(self.ax_3d, 'U',
                         np.array([1, 0, 0]),
                         np.array([0, 1, 0]),
                         np.array([0, 0, 1]))

        # D (yellow) - at x=0 (appears on visual right due to view angle)
        self.draw_face_3d(self.ax_3d, 'D',
                         np.array([0, 1, 0]),
                         np.array([0, -1, 0]),
                         np.array([0, 0, 1]))

        # Set axis properties
        self.ax_3d.set_xlim([-0.1, 1.1])
        self.ax_3d.set_ylim([-0.1, 1.1])
        self.ax_3d.set_zlim([-0.1, 1.1])
        self.ax_3d.set_box_aspect([1, 1, 1])
        self.ax_3d.axis('off')
        self.ax_3d.view_init(elev=self.elev, azim=self.azim)

        # Add orientation labels matching physical cube (visual right is x=0 due to view angle)
        self.ax_3d.text(0.5, 0.5, 1.15, 'BLUE (Top)', ha='center', fontsize=9, fontweight='bold', color='blue')
        self.ax_3d.text(0.5, 1.15, 0.5, 'RED (Front)', ha='center', fontsize=9, fontweight='bold', color='red')
        self.ax_3d.text(-0.15, 0.5, 0.5, 'YELLOW (Right)', ha='center', fontsize=9, fontweight='bold', color='#DAA520')

    def draw_cube_body(self):
        """Draw black cube interior for depth"""
        # Just draw thin black edges
        edges = [
            # Bottom face edges
            ([0, 1], [0, 0], [0, 0]),
            ([0, 0], [0, 0], [0, 1]),
            ([1, 1], [0, 0], [0, 1]),
            ([0, 1], [0, 0], [1, 1]),
            # Top face edges
            ([0, 1], [1, 1], [0, 0]),
            ([0, 0], [1, 1], [0, 1]),
            ([1, 1], [1, 1], [0, 1]),
            ([0, 1], [1, 1], [1, 1]),
            # Vertical edges
            ([0, 0], [0, 1], [0, 0]),
            ([1, 1], [0, 1], [0, 0]),
            ([0, 0], [0, 1], [1, 1]),
            ([1, 1], [0, 1], [1, 1]),
        ]
        for xs, ys, zs in edges:
            self.ax_3d.plot(xs, ys, zs, 'k-', linewidth=2)

    def draw_unfolded_2d(self):
        """Draw 2D unfolded cube view"""
        self.ax_2d.clear()
        self.ax_2d.set_xlim(-0.3, 4.3)
        self.ax_2d.set_ylim(-0.5, 3.5)
        self.ax_2d.set_aspect('equal')
        self.ax_2d.axis('off')
        self.ax_2d.set_title('Hold cube: BLUE top, RED facing you', fontsize=10, fontweight='bold')

        # Face positions in unfolded view (physical cube: blue top, red front):
        #       [F] (Blue=Top)
        #   [U] [R] [D] [L]
        #       [B] (Green=Bottom)
        # Where: U=White(Left), R=Red(Front), D=Yellow(Right), L=Orange(Back)
        face_positions = {
            'F': (1, 2),  # Blue - Top
            'U': (0, 1),  # White - Left
            'R': (1, 1),  # Red - Front
            'D': (2, 1),  # Yellow - Right
            'L': (3, 1),  # Orange - Back
            'B': (1, 0)   # Green - Bottom
        }

        # Face labels with center colors (physical orientation)
        face_labels = {
            'F': 'TOP\n(Blue)',
            'B': 'BOTTOM\n(Green)',
            'R': 'FRONT\n(Red)',
            'L': 'BACK\n(Orange)',
            'D': 'RIGHT\n(Yellow)',
            'U': 'LEFT\n(White)'
        }

        for face_name, (fx, fy) in face_positions.items():
            face = self.faces[face_name]

            # Draw face background
            bg_color = '#f0f0f0' if face_name != self.current_face else '#ffffcc'
            rect = plt.Rectangle((fx, fy), 1, 1, facecolor=bg_color, edgecolor='#808080', linewidth=2)
            self.ax_2d.add_patch(rect)

            for i in range(3):
                for j in range(3):
                    x = fx + j/3 + 0.02
                    y = fy + (2-i)/3 + 0.02
                    w = 1/3 - 0.04
                    h = 1/3 - 0.04

                    color = self.color_map[face[i][j]]
                    edge_color = '#404040'
                    line_width = 1

                    # Highlight selected cell
                    if face_name == self.current_face and i == self.current_row and j == self.current_col:
                        edge_color = '#00FFFF'
                        line_width = 3

                    # Mark center as special
                    if i == 1 and j == 1:
                        edge_color = '#FFD700'
                        line_width = 2

                    rect = plt.Rectangle((x, y), w, h,
                                         facecolor=color, edgecolor=edge_color,
                                         linewidth=line_width)
                    self.ax_2d.add_patch(rect)

            # Add face label with center color
            label = face_labels[face_name]
            self.ax_2d.text(fx + 0.5, fy + 1.12, label,
                           ha='center', va='bottom', fontsize=8, fontweight='bold',
                           color='#0066cc' if face_name == self.current_face else '#666666')

    def draw_color_palette(self):
        """Draw color selection palette"""
        self.ax_palette.clear()
        self.ax_palette.set_xlim(0, 6)
        self.ax_palette.set_ylim(0, 1.5)
        self.ax_palette.set_aspect('equal')
        self.ax_palette.axis('off')
        self.ax_palette.set_title('Color Palette (Click or press 1-6)', fontsize=10, fontweight='bold')

        for i, color in enumerate(self.colors):
            x = i
            edge_color = '#00FFFF' if color == self.selected_color else '#404040'
            line_width = 3 if color == self.selected_color else 1

            rect = plt.Rectangle((x + 0.1, 0.1), 0.8, 0.8,
                                 facecolor=self.color_map[color],
                                 edgecolor=edge_color,
                                 linewidth=line_width)
            self.ax_palette.add_patch(rect)

            # Label with key number
            self.ax_palette.text(x + 0.5, 1.1, f'{i+1}', ha='center', fontsize=9, fontweight='bold')
            self.ax_palette.text(x + 0.5, -0.15, color[0].upper(), ha='center', fontsize=8)

    def draw_info_panel(self):
        """Draw information panel"""
        self.ax_info.clear()
        self.ax_info.set_xlim(0, 10)
        self.ax_info.set_ylim(0, 10)
        self.ax_info.axis('off')

        y = 9.5

        # Current selection info
        self.ax_info.text(0, y, f"Selected: {self.face_names[self.current_face]}",
                         fontsize=10, fontweight='bold', color='#0066cc')
        y -= 0.8
        self.ax_info.text(0, y, f"Position: Row {self.current_row+1}, Col {self.current_col+1}", fontsize=9)
        y -= 1.0

        # Guided mode indicator
        if self.guided_mode:
            self.ax_info.text(0, y, "GUIDED MODE ON", fontsize=10, fontweight='bold', color='green')
            y -= 0.7
            progress = sum(1 for f in self.face_order[:self.guided_face_index]) * 8
            total = 48
            self.ax_info.text(0, y, f"Progress: {progress}/{total} stickers", fontsize=9)
            y -= 1.0

        # Solver mode indicator
        mode = self.solve_modes[self.solve_mode_index]
        mode_label = self.solve_mode_labels[mode]
        self.ax_info.text(0, y, f"Mode: {mode_label}",
                         fontsize=10, fontweight='bold', color='#cc6600')
        y -= 0.6
        self.ax_info.text(0, y, "(press M to cycle modes)", fontsize=8, color='gray')
        y -= 0.8

        # Last solve stats
        if hasattr(self, 'last_solve_info') and self.last_solve_info:
            info = self.last_solve_info
            self.ax_info.text(0, y, f"Last: {info['moves']} moves, {info['time_ms']:.0f} ms",
                             fontsize=9, color='#006600')
            y -= 0.8

        y -= 0.3
        # Controls
        controls = [
            "CONTROLS:",
            "Click: Select sticker/color",
            "Drag: Rotate 3D view",
            "Tab/Shift+Tab: Next/Prev face",
            "Arrow keys: Move in face",
            "1-6: Set color",
            "M: Cycle solver mode",
            "Enter: SOLVE",
            "N: Next solution step",
            "S: Random scramble",
            "R: Reset | G: Guide mode",
            "Q: Quit"
        ]

        for text in controls:
            self.ax_info.text(0, y, text, fontsize=8, family='monospace')
            y -= 0.65

        # Message display
        if self.message:
            y -= 0.5
            self.ax_info.text(0, y, self.message, fontsize=9, fontweight='bold',
                             color=self.message_color, wrap=True)

    def draw_all(self):
        """Draw all components"""
        self.draw_cube_3d()
        self.draw_unfolded_2d()
        self.draw_color_palette()
        self.draw_info_panel()
        self.fig.canvas.draw_idle()

    def rotate_face_cw(self, face):
        """Rotate a 3x3 face 90 degrees clockwise"""
        return [[face[2-j][i] for j in range(3)] for i in range(3)]

    def rotate_face_ccw(self, face):
        """Rotate a 3x3 face 90 degrees counterclockwise"""
        return [[face[j][2-i] for j in range(3)] for i in range(3)]

    def rotate_face_180(self, face):
        """Rotate a 3x3 face 180 degrees"""
        return [[face[2-i][2-j] for j in range(3)] for i in range(3)]

    def flip_horizontal(self, face):
        """Flip a 3x3 face horizontally"""
        return [[face[i][2-j] for j in range(3)] for i in range(3)]

    def flip_vertical(self, face):
        """Flip a 3x3 face vertically"""
        return [[face[2-i][j] for j in range(3)] for i in range(3)]

    def to_kociemba_string(self):
        """Convert current visual cube state to Kociemba's facelet string.

        We **respect your physical orientation**:
        - BLUE center = TOP
        - RED center  = FRONT
        - YELLOW      = RIGHT
        - GREEN       = BOTTOM
        - WHITE       = LEFT
        - ORANGE      = BACK

        Instead of hard‑coding color→letter (like white→U, yellow→D, ...),
        we:
        - Read the six face *centres* the user actually set.
        - Assign letters U,R,F,D,L,B to those centre colors according to
          their physical positions (Top, Right, Front, Bottom, Left, Back).
        - Map every sticker of a given centre color to that face letter.
        """
        physical_faces = self._physical_faces()

        # Build dynamic color→letter mapping from the six centres.
        color_to_letter = {}
        self.letter_to_color = {}
        for face_name, letter in physical_faces:
            center_color = self.faces[face_name][1][1]
            color_to_letter[center_color] = letter
            self.letter_to_color[letter] = center_color

        # Emit stickers in Kociemba physical face order:
        #   Top, Right, Front, Bottom, Left, Back.
        result = ""
        for face_name, _letter in physical_faces:
            face = self.faces[face_name]
            for row in face:
                for color in row:
                    result += color_to_letter.get(color, '?')
        return result

    def solve(self):
        """Solve the cube using the currently selected mode and return solution"""
        cube_string = self.to_kociemba_string()

        # Check color counts
        counts = {c: cube_string.count(c) for c in 'URFDLB'}

        if any(v != 9 for v in counts.values()):
            bad_colors = [c for c, v in counts.items() if v != 9]
            return None, f"Color count error: {', '.join(f'{c}={counts[c]}' for c in bad_colors)}"

        mode = self.solve_modes[self.solve_mode_index]
        mode_label = self.solve_mode_labels[mode]

        try:
            import time as _time
            t0 = _time.time()
            solution = solve_cube_compat(cube_string, mode=mode)
            elapsed_ms = (_time.time() - t0) * 1000

            # Store parsed moves for step-by-step guidance
            self.solution_moves = solution.split()
            self.current_move_index = -1  # not started yet
            self.base_cube_string = cube_string
            self.playback_cube_string = cube_string
            self.last_solve_info = {
                'mode': mode_label,
                'moves': len(self.solution_moves),
                'time_ms': elapsed_ms,
                'solution': solution,
            }
            return solution, None
        except ValueError as e:
            # Clear any previous solution
            self.solution_moves = []
            self.current_move_index = -1
            self.base_cube_string = None
            self.playback_cube_string = None
            self.last_solve_info = None
            return None, str(e)

    # Demo scrambles with known K=1 vs K=5 deltas, reproduced from the
    # seed-42 benchmark (results/benchmark_200_retrained.json).  Each tuple is
    # (facelet_string, expected_k1_moves, expected_k5_moves).
    DEMO_SCRAMBLES = [
        ('DBRLUDBFBDRFDRLDUUUURUFFRRFFFLLDRBBLBFLLLBUUUDRLBBDFDR', 21, 18),
        ('FDFDUBUFDBRDBRRRFDLULRFUFUDLLBDDLUBFULBRLFRBURFRUBDLLB', 21, 18),
        ('DLLDULRUFLUBRRFLDDDBUDFFFRUUBBBDBURFRRFULFRLRDDBLBFLUB', 22, 20),
        ('DDURUFDFFUUBLRDFUFRDRLFFUBUBRLBDRDDLFFBULULLRLRRLBBDBB', 22, 20),
    ]

    def load_demo_scramble(self, index: int):
        """Load one of the preset demo scrambles with a known K=1 vs K=5 delta."""
        if index < 0 or index >= len(self.DEMO_SCRAMBLES):
            return
        cube_string = self.DEMO_SCRAMBLES[index][0]

        # Reset centers so the letter_to_color mapping is right.
        self.faces = {
            'U': [['white']*3 for _ in range(3)],
            'D': [['yellow']*3 for _ in range(3)],
            'F': [['blue']*3 for _ in range(3)],
            'B': [['green']*3 for _ in range(3)],
            'R': [['red']*3 for _ in range(3)],
            'L': [['orange']*3 for _ in range(3)]
        }
        self._update_faces_from_cube_string(cube_string)

        # Clear any previous solve state.
        self.solution_moves = []
        self.current_move_index = -1
        self.base_cube_string = None
        self.playback_cube_string = None

        self.message = "Random scramble applied. Press Enter to solve!"
        self.message_color = 'blue'

    def randomize(self):
        """Generate a valid random cube state by applying random moves to a solved cube."""
        # Reset to solved state first to ensure correct center mapping
        self.faces = {
            'U': [['white']*3 for _ in range(3)],
            'D': [['yellow']*3 for _ in range(3)],
            'F': [['blue']*3 for _ in range(3)],
            'B': [['green']*3 for _ in range(3)],
            'R': [['red']*3 for _ in range(3)],
            'L': [['orange']*3 for _ in range(3)]
        }

        # Build letter_to_color mapping from solved centers
        solved = self.to_kociemba_string()

        # Apply 20-30 random moves to scramble
        move_faces = ['U', 'R', 'F', 'D', 'L', 'B']
        suffixes = ['', "'", '2']

        cube_string = solved
        num_moves = np.random.randint(20, 31)
        last_face = None
        for _ in range(num_moves):
            available = [m for m in move_faces if m != last_face]
            face = available[np.random.randint(len(available))]
            suffix = suffixes[np.random.randint(len(suffixes))]
            move = face + suffix
            cube_string = self._apply_kociemba_move_to_string(cube_string, move)
            last_face = face

        # Update the UI faces from the scrambled state
        self._update_faces_from_cube_string(cube_string)

        # Clear any previous solution
        self.solution_moves = []
        self.current_move_index = -1
        self.base_cube_string = None
        self.playback_cube_string = None

        self.message = f"Random scramble applied ({num_moves} moves). Press Enter to solve!"
        self.message_color = 'blue'

    def set_sticker_color(self, face, row, col, color):
        """Set a sticker's color (except centers)"""
        if row == 1 and col == 1:
            self.message = "Cannot change center piece!"
            self.message_color = 'red'
            return False

        self.faces[face][row][col] = color
        self.message = ""
        return True

    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes == self.ax_palette:
            # Click on color palette
            x = event.xdata
            if x is not None:
                color_idx = int(x)
                if 0 <= color_idx < 6:
                    self.selected_color = self.colors[color_idx]
                    # Also set the current sticker to this color
                    self.set_sticker_color(self.current_face, self.current_row,
                                          self.current_col, self.selected_color)
                    if self.guided_mode:
                        self.advance_guided()
                    self.draw_all()

        elif event.inaxes == self.ax_2d:
            # Click on 2D unfolded view
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # Determine which face and cell was clicked
                face_positions = {
                    'F': (1, 2), 'U': (0, 1), 'R': (1, 1),
                    'D': (2, 1), 'L': (3, 1), 'B': (1, 0)
                }

                for face_name, (fx, fy) in face_positions.items():
                    if fx <= x < fx + 1 and fy <= y < fy + 1:
                        col = int((x - fx) * 3)
                        row = 2 - int((y - fy) * 3)
                        col = max(0, min(2, col))
                        row = max(0, min(2, row))

                        self.current_face = face_name
                        self.current_row = row
                        self.current_col = col
                        self.draw_all()
                        break

    def on_press(self, event):
        """Handle mouse button press for dragging"""
        if event.inaxes == self.ax_3d:
            self.dragging = True
            self.last_mouse_pos = (event.x, event.y)  # Use pixel coordinates

    def on_release(self, event):
        """Handle mouse button release"""
        self.dragging = False
        self.last_mouse_pos = None

    def on_motion(self, event):
        """Handle mouse motion for rotation"""
        if self.dragging and self.last_mouse_pos:
            if event.x is not None and event.y is not None:
                dx = event.x - self.last_mouse_pos[0]
                dy = event.y - self.last_mouse_pos[1]

                # Use pixel-based rotation (0.3 degrees per pixel for smooth control)
                self.azim -= dx * 0.3
                self.elev -= dy * 0.3
                self.elev = max(-90, min(90, self.elev))

                self.last_mouse_pos = (event.x, event.y)
                self.draw_cube_3d()
                self.fig.canvas.draw_idle()

    def advance_guided(self):
        """Advance to next position in guided mode"""
        self.guided_position += 1
        if self.guided_position == 4:  # Skip center
            self.guided_position = 5

        if self.guided_position >= 9:
            self.guided_position = 0
            self.guided_face_index += 1

            if self.guided_face_index >= 6:
                self.guided_mode = False
                self.message = "All faces complete! Press Enter to solve."
                self.message_color = 'green'
                return

        self.current_face = self.face_order[self.guided_face_index]
        self.current_row = self.guided_position // 3
        self.current_col = self.guided_position % 3

    def on_key(self, event):
        """Handle keyboard input"""
        if event.key == 'q':
            plt.close(self.fig)
            return

        if event.key == 'up':
            self.current_row = (self.current_row - 1) % 3
            if self.current_row == 1 and self.current_col == 1:
                self.current_row = 0
        elif event.key == 'down':
            self.current_row = (self.current_row + 1) % 3
            if self.current_row == 1 and self.current_col == 1:
                self.current_row = 2
        elif event.key == 'left':
            self.current_col = (self.current_col - 1) % 3
            if self.current_row == 1 and self.current_col == 1:
                self.current_col = 0
        elif event.key == 'right':
            self.current_col = (self.current_col + 1) % 3
            if self.current_row == 1 and self.current_col == 1:
                self.current_col = 2
        elif event.key == 'tab':
            faces = ['U', 'R', 'F', 'D', 'L', 'B']
            idx = faces.index(self.current_face)
            self.current_face = faces[(idx + 1) % 6]
            self.current_row = 0
            self.current_col = 0
        elif event.key == 'shift+tab':
            faces = ['U', 'R', 'F', 'D', 'L', 'B']
            idx = faces.index(self.current_face)
            self.current_face = faces[(idx - 1) % 6]
            self.current_row = 0
            self.current_col = 0
        elif event.key in ['1', '2', '3', '4', '5', '6']:
            color_idx = int(event.key) - 1
            self.selected_color = self.colors[color_idx]
            if self.set_sticker_color(self.current_face, self.current_row,
                                      self.current_col, self.selected_color):
                if self.guided_mode:
                    self.advance_guided()
        elif event.key == 'r':
            # Reset to solved state
            self.faces = {
                'U': [['white']*3 for _ in range(3)],
                'D': [['yellow']*3 for _ in range(3)],
                'F': [['blue']*3 for _ in range(3)],
                'B': [['green']*3 for _ in range(3)],
                'R': [['red']*3 for _ in range(3)],
                'L': [['orange']*3 for _ in range(3)]
            }
            self.guided_mode = False
            self.guided_face_index = 0
            self.guided_position = 0
            self.message = "Cube reset to solved state"
            self.message_color = 'blue'
            self.solution_moves = []
            self.current_move_index = -1
            self.base_cube_string = None
            self.playback_cube_string = None
        elif event.key == 's':
            # Generate random scramble
            self.randomize()
        elif event.key == 'g':
            # Toggle guided mode
            self.guided_mode = not self.guided_mode
            if self.guided_mode:
                self.guided_face_index = 0
                self.guided_position = 0
                self.current_face = 'U'
                self.current_row = 0
                self.current_col = 0
                self.message = "Guided mode: Enter colors face by face"
                self.message_color = 'green'
            else:
                self.message = "Guided mode disabled"
                self.message_color = 'gray'
        elif event.key == 'n':
            # Step through solution in clear words
            if not self.solution_moves:
                self.message = "No solution loaded yet. Press Enter to solve first."
                self.message_color = 'red'
            else:
                # Initialize playback from the exact cube state we solved.
                if not self.playback_cube_string:
                    self.playback_cube_string = self.base_cube_string or self.to_kociemba_string()

                # Advance move index (clamp at end)
                next_index = min(self.current_move_index + 1, len(self.solution_moves) - 1)
                move = self.solution_moves[next_index]

                # Apply the move to the displayed cube state
                self.playback_cube_string = self._apply_kociemba_move_to_string(self.playback_cube_string, move)
                self._update_faces_from_cube_string(self.playback_cube_string)

                self.current_move_index = next_index
                step_num = self.current_move_index + 1
                total = len(self.solution_moves)

                # Human-readable instruction (no U / U' etc.)
                letter = move[0]
                suffix = move[1:] if len(move) > 1 else ''
                pos_names = {'U': 'top', 'R': 'right', 'F': 'front', 'D': 'bottom', 'L': 'left', 'B': 'back'}
                color = getattr(self, 'letter_to_color', {}).get(letter)
                if not color:
                    # Make sure mapping exists
                    _ = self.to_kociemba_string()
                    color = getattr(self, 'letter_to_color', {}).get(letter, 'unknown')

                direction = "90 degrees clockwise"
                if suffix == "'":
                    direction = "90 degrees counter‑clockwise"
                elif suffix == "2":
                    direction = "180 degrees (half‑turn)"

                position = pos_names.get(letter, 'face')
                self.message = f"Step {step_num}/{total}: Turn the {color.upper()} ({position}) face {direction}."
                self.message_color = 'green'

                if self.current_move_index == total - 1:
                    self.message += " Done — cube on screen should now be solved."

        elif event.key == 'm':
            # Cycle solver mode
            self.solve_mode_index = (self.solve_mode_index + 1) % len(self.solve_modes)
            mode = self.solve_modes[self.solve_mode_index]
            label = self.solve_mode_labels[mode]
            self.message = f"Solver mode: {label}"
            self.message_color = '#0066cc'
            print(f"Solver mode changed to: {label}")

        elif event.key == 'enter':
            if self.solve_in_progress:
                # A solve is already running — let the timer drive the UI.
                return

            mode = self.solve_modes[self.solve_mode_index]
            if mode != 'baseline':
                # Slow modes: run on a background thread with a progress bar.
                self._start_async_solve()
                return

            mode_label = self.solve_mode_labels[mode]
            self.message = f"Solving with {mode_label}..."
            self.message_color = 'orange'
            self.draw_all()  # show "Solving..." before blocking call

            solution, error = self.solve()
            if error:
                self.message = f"Error: {error}"
                self.message_color = 'red'
                print(f"Error: {error}")
            else:
                if solution:
                    info = self.last_solve_info
                    self.message = (
                        f"[{info['mode']}] {info['moves']} moves in {info['time_ms']:.0f} ms. "
                        f"Press 'N' for step-by-step."
                    )
                    self.message_color = 'green'
                    print(f"\nSOLUTION FOUND! [{info['mode']}]")
                    print(f"Moves: {solution}")
                    print(f"Total moves: {info['moves']} | Time: {info['time_ms']:.1f} ms")
                else:
                    self.message = "Cube is already solved!"
                    self.message_color = 'green'

        self.draw_all()

    def _start_async_solve(self):
        """Run the solver on a background thread with an animated progress bar."""
        cube_string = self.to_kociemba_string()

        counts = {c: cube_string.count(c) for c in 'URFDLB'}
        if any(v != 9 for v in counts.values()):
            bad_colors = [c for c, v in counts.items() if v != 9]
            self.message = (
                "Color count error: "
                + ", ".join(f"{c}={counts[c]}" for c in bad_colors)
            )
            self.message_color = 'red'
            self.draw_all()
            return

        mode = self.solve_modes[self.solve_mode_index]
        mode_label = self.solve_mode_labels[mode]

        self.solve_in_progress = True
        self.solve_result = None
        self.solve_start_time = time.time()
        self.progress_position = 0
        self._pending_cube_string = cube_string

        # Build the matplotlib state for the entire UI plus an initial bar
        # frame, then force a synchronous render BEFORE the solver thread
        # starts so the bar is already on screen when the GIL contention kicks
        # in.
        self.message = f"Solving with {mode_label}..."
        self.message_color = 'orange'
        self.draw_all()
        self._update_progress_bar()

        # Capture the static background of the progress-bar axes (track only,
        # without the marquee/text) so each tick can do a tiny blit instead of
        # redrawing the whole figure.
        self.progress_marquee.set_visible(False)
        self.progress_text.set_visible(False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        try:
            self._progress_bg = self.fig.canvas.copy_from_bbox(self.ax_progress.bbox)
        except Exception:
            self._progress_bg = None
        self.progress_marquee.set_visible(True)
        self.progress_text.set_visible(True)
        if self._progress_bg is not None:
            self.fig.canvas.restore_region(self._progress_bg)
            self.ax_progress.draw_artist(self.progress_marquee)
            self.ax_progress.draw_artist(self.progress_text)
            self.fig.canvas.blit(self.ax_progress.bbox)
            self.fig.canvas.flush_events()
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        # Make Python's thread switch much more frequent than the default 5 ms
        # so the matplotlib Tk timer can actually fire while the pure-Python
        # solver thread is hammering the GIL.
        self._original_switch_interval = _sys.getswitchinterval()
        _sys.setswitchinterval(0.001)

        def worker():
            try:
                t0 = time.time()
                solution = solve_cube_compat(cube_string, mode=mode)
                elapsed_ms = (time.time() - t0) * 1000
                self.solve_result = ('ok', solution, elapsed_ms)
            except Exception as e:
                self.solve_result = ('err', str(e), 0)

        self.solve_thread = threading.Thread(target=worker, daemon=True)
        self.solve_thread.start()

        if self.progress_timer is None:
            self.progress_timer = self.fig.canvas.new_timer(interval=50)
            self.progress_timer.add_callback(self._on_progress_tick)
        self.progress_timer.start()

    def _on_progress_tick(self):
        """Timer callback: advance the marquee or finalize when the thread completes."""
        if not self.solve_in_progress:
            if self.progress_timer is not None:
                self.progress_timer.stop()
            return

        if self.solve_result is not None:
            if self.progress_timer is not None:
                self.progress_timer.stop()
            self._finalize_solve_result()
            return

        self.progress_position += 2
        self._update_progress_bar()

        # Cheap partial repaint: restore the static background of the bar
        # axes, redraw only the marquee + elapsed-time text, blit just that
        # region. This stays fast even while the solver is busy.
        if self._progress_bg is not None:
            self.fig.canvas.restore_region(self._progress_bg)
            self.ax_progress.draw_artist(self.progress_marquee)
            self.ax_progress.draw_artist(self.progress_text)
            self.fig.canvas.blit(self.ax_progress.bbox)
        else:
            self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_progress_bar(self):
        """Advance the marquee bar and refresh the elapsed-time label."""
        if self.ax_progress is None:
            return

        if not self.solve_in_progress:
            self.ax_progress.set_visible(False)
            return

        self.ax_progress.set_visible(True)
        track_width = 100
        bar_width = 25
        span = (track_width - bar_width) * 2
        pos = self.progress_position % span
        x = pos if pos <= track_width - bar_width else span - pos
        self.progress_marquee.set_x(x)

        elapsed = time.time() - self.solve_start_time
        mode_label = self.solve_mode_labels[self.solve_modes[self.solve_mode_index]]
        self.progress_text.set_text(
            f'Solving... [{mode_label}]   {elapsed:0.1f}s'
        )

    def _finalize_solve_result(self):
        """Apply the worker thread's result to the UI state."""
        # Restore the default Python thread switch interval.
        if self._original_switch_interval is not None:
            try:
                _sys.setswitchinterval(self._original_switch_interval)
            except Exception:
                pass
            self._original_switch_interval = None

        self._progress_bg = None

        result = self.solve_result
        self.solve_result = None
        self.solve_in_progress = False
        if self.ax_progress is not None:
            self.ax_progress.set_visible(False)

        if not result:
            self.draw_all()
            return

        status = result[0]
        if status == 'err':
            self.solution_moves = []
            self.current_move_index = -1
            self.base_cube_string = None
            self.playback_cube_string = None
            self.last_solve_info = None
            self.message = f"Error: {result[1]}"
            self.message_color = 'red'
            print(f"Error: {result[1]}")
        else:
            solution = result[1]
            elapsed_ms = result[2]
            cube_string = self._pending_cube_string
            mode_label = self.solve_mode_labels[self.solve_modes[self.solve_mode_index]]

            self.solution_moves = solution.split() if solution else []
            self.current_move_index = -1
            self.base_cube_string = cube_string
            self.playback_cube_string = cube_string
            self.last_solve_info = {
                'mode': mode_label,
                'moves': len(self.solution_moves),
                'time_ms': elapsed_ms,
                'solution': solution,
            }

            if solution:
                info = self.last_solve_info
                self.message = (
                    f"[{info['mode']}] {info['moves']} moves in {info['time_ms']:.0f} ms. "
                    f"Press 'N' for step-by-step."
                )
                self.message_color = 'green'
                print(f"\nSOLUTION FOUND! [{info['mode']}]")
                print(f"Moves: {solution}")
                print(f"Total moves: {info['moves']} | Time: {info['time_ms']:.1f} ms")
            else:
                self.message = "Cube is already solved!"
                self.message_color = 'green'

        self.draw_all()

    def setup_figure(self):
        """Setup the matplotlib figure with all subplots"""
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('3D Rubik\'s Cube Input - Easy Color Entry')

        # Create grid layout
        gs = self.fig.add_gridspec(3, 3, width_ratios=[1.2, 1.2, 0.8],
                                   height_ratios=[1, 0.25, 0.05],
                                   hspace=0.3, wspace=0.2)

        # 3D cube view
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d')

        # 2D unfolded view
        self.ax_2d = self.fig.add_subplot(gs[0, 1])

        # Info panel
        self.ax_info = self.fig.add_subplot(gs[0, 2])

        # Color palette (spans bottom)
        self.ax_palette = self.fig.add_subplot(gs[1, :2])

        # Demo scramble buttons (preset hard cubes where K=5 beats K=1).
        # Two rows of two buttons above the Random Scramble button.
        demo_positions = [
            (0.82,  0.135),
            (0.87,  0.135),
            (0.82,  0.095),
            (0.87,  0.095),
        ]
        self.demo_buttons = []
        for i, (dx, dy) in enumerate(demo_positions):
            ax_demo = self.fig.add_axes([dx, dy, 0.048, 0.035])
            btn_demo = Button(ax_demo, f'Demo {i+1}',
                              color='#FFF0C8', hovercolor='#FFE090')
            btn_demo.label.set_fontsize(7)
            # Capture i via default arg so the closure binds the correct index.
            btn_demo.on_clicked(lambda event, idx=i: (self.load_demo_scramble(idx), self.draw_all()))
            self.demo_buttons.append(btn_demo)

        # Add small random scramble button at the bottom-right
        ax_btn_random = self.fig.add_axes([0.82, 0.05, 0.1, 0.04])
        self.btn_random = Button(ax_btn_random, 'Random Scramble',
                                 color='#E8E8FF', hovercolor='#C0C0FF')
        ax_btn_random.text(0.5, 0.5, '', fontsize=8, ha='center', va='center')
        self.btn_random.label.set_fontsize(8)
        self.btn_random.on_clicked(lambda event: (self.randomize(), self.draw_all()))

        # Progress bar (hidden until an async solve starts)
        self.ax_progress = self.fig.add_axes([0.1, 0.945, 0.55, 0.03])
        self.ax_progress.set_xlim(0, 100)
        self.ax_progress.set_ylim(0, 1)
        self.ax_progress.axis('off')
        self.ax_progress.set_visible(False)
        self.progress_track = mpatches.Rectangle(
            (0, 0), 100, 1,
            facecolor='#E8E8E8', edgecolor='#808080', linewidth=1,
        )
        self.ax_progress.add_patch(self.progress_track)
        self.progress_marquee = mpatches.Rectangle(
            (0, 0), 25, 1,
            facecolor='#FF8800', edgecolor='none',
        )
        self.ax_progress.add_patch(self.progress_marquee)
        self.progress_text = self.ax_progress.text(
            50, 0.5, '',
            ha='center', va='center',
            fontsize=9, fontweight='bold', color='white',
        )

        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Instructions at bottom
        self.fig.text(0.5, 0.02,
                     'Physical cube: BLUE top, RED front, YELLOW right. Match your cube to this orientation.',
                     ha='center', fontsize=10, style='italic', color='#666666')

    def run(self):
        """Run the interactive 3D cube input"""
        self.setup_figure()
        self.message = "Click stickers to select, then click a color. Press G for guided mode."
        self.message_color = '#666666'
        self.draw_all()
        plt.show()


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("  3D RUBIK'S CUBE INPUT INTERFACE")
    print("="*60)
    print("\nThis tool helps you input your cube state visually.")
    print("\nOrientation Guide (Physical Cube):")
    print("  - Hold the cube with BLUE center facing UP (top)")
    print("  - RED center should face toward you (front)")
    print("  - YELLOW will be on your RIGHT")
    print("  - WHITE on LEFT, ORANGE in BACK, GREEN on BOTTOM")
    print("\nTips:")
    print("  - Press 'G' for guided mode (step-by-step)")
    print("  - Click on the 2D view to select stickers")
    print("  - Drag on the 3D view to rotate")
    print("  - Press Enter when done to solve")
    print("="*60 + "\n")

    cube = RubiksCube3D()
    cube.run()


if __name__ == "__main__":
    main()
