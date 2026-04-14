import streamlit as st
import time
import random
import sys
from pathlib import Path

# Ensure the local enhanced kociemba is found before the pip version
sys.path.insert(0, str(Path(__file__).resolve().parent / 'kociemba'))
from kociemba import solve as kociemba_solve
from kociemba.pykociemba import tools

# ── Constants ──────────────────────────────────────────────────────

COLOR_MAP = {
    'W': ('white',   '#FFFFFF', 'U'),
    'R': ('red',     '#E80000', 'R'),
    'B': ('blue',    '#0051FF', 'F'),
    'Y': ('yellow',  '#FFD500', 'D'),
    'O': ('orange',  '#FF8C00', 'L'),
    'G': ('green',   '#00C800', 'B'),
}

# Display label -> (short key, kociemba letter)
COLOR_OPTIONS = ['W', 'R', 'B', 'Y', 'O', 'G']

FACES = [
    ('U', 'Up',    'W', 'White'),
    ('R', 'Right', 'R', 'Red'),
    ('F', 'Front', 'B', 'Blue'),
    ('D', 'Down',  'Y', 'Yellow'),
    ('L', 'Left',  'O', 'Orange'),
    ('B', 'Back',  'G', 'Green'),
]

SOLVER_MODES = {
    'Baseline (K=1)': {
        'desc': 'Original Kociemba — tries one Phase 1 path',
        'kwargs': dict(max_phase1_solutions=1),
    },
    'Multi-Solution (K=5)': {
        'desc': 'Tries 5 Phase 1 paths, keeps the shortest solution',
        'kwargs': dict(max_phase1_solutions=5, timeout=5000),
    },
    'Anytime (K=5, 2s)': {
        'desc': 'Returns the best solution found within 2 seconds',
        'kwargs': dict(max_phase1_solutions=5, time_budget_sec=2.0),
    },
    'Neural LUT (K=5)': {
        'desc': 'K=5 + neural move ordering via precomputed lookup table',
        'kwargs': dict(max_phase1_solutions=5, timeout=5000,
                       neural_strategy='lut'),
    },
}


def solve_with_mode(cube_string, mode_name):
    """Solve cube using the selected mode. Returns (solution, elapsed_ms)."""
    kwargs = SOLVER_MODES[mode_name]['kwargs']
    t0 = time.time()
    solution = kociemba_solve(cube_string, **kwargs)
    elapsed = (time.time() - t0) * 1000
    return solution, elapsed


def cube_state_to_string(state):
    """Convert session state cube dict to 54-char Kociemba string."""
    result = ''
    for face_key, _, _, _ in FACES:
        for c in state[face_key]:
            result += COLOR_MAP[c][2]
    return result


def validate_cube_string(s):
    """Check each Kociemba letter appears exactly 9 times."""
    for letter in 'URFDLB':
        if s.count(letter) != 9:
            return False
    return True


def random_scramble_state():
    """Generate a random valid cube and return it as our color-key state dict."""
    cube_str = tools.randomCube()
    # Map kociemba letters back to our color keys
    letter_to_color = {v[2]: k for k, v in COLOR_MAP.items()}
    state = {}
    idx = 0
    for face_key, _, _, _ in FACES:
        face_colors = []
        for i in range(9):
            face_colors.append(letter_to_color[cube_str[idx]])
            idx += 1
        state[face_key] = face_colors
    return state


def render_face_grid(colors, face_key, editable=True):
    """Render a 3x3 face grid with color pickers."""
    for row in range(3):
        cols = st.columns(3)
        for col in range(3):
            idx = row * 3 + col
            c = colors[idx]
            _, hex_color, _ = COLOR_MAP[c]

            with cols[col]:
                if idx == 4 and editable:
                    # Center is locked
                    st.markdown(
                        f'<div style="width:100%;aspect-ratio:1;background:{hex_color};'
                        f'border:3px solid #333;border-radius:4px;display:flex;'
                        f'align-items:center;justify-content:center;font-size:10px;'
                        f'color:#333;font-weight:bold;">CENTER</div>',
                        unsafe_allow_html=True,
                    )
                elif editable:
                    new_c = st.selectbox(
                        ' ', COLOR_OPTIONS,
                        index=COLOR_OPTIONS.index(c),
                        key=f'sel_{face_key}_{idx}',
                        label_visibility='collapsed',
                    )
                    if new_c != c:
                        st.session_state.cube[face_key][idx] = new_c
                        st.rerun()
                else:
                    st.markdown(
                        f'<div style="width:100%;aspect-ratio:1;background:{hex_color};'
                        f'border:1px solid #555;border-radius:3px;"></div>',
                        unsafe_allow_html=True,
                    )


def render_mini_face(colors):
    """Render a tiny non-interactive face preview."""
    html = '<div style="display:grid;grid-template-columns:repeat(3,18px);gap:1px;">'
    for c in colors:
        _, hex_color, _ = COLOR_MAP[c]
        html += f'<div style="width:18px;height:18px;background:{hex_color};border:1px solid #444;border-radius:2px;"></div>'
    html += '</div>'
    return html


# ── Page Config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Rubik's Cube Solver — CIIR 2026",
    page_icon="🧊",
    layout="wide",
)

# ── Session State Init ─────────────────────────────────────────────

if 'cube' not in st.session_state:
    st.session_state.cube = {
        'U': ['W'] * 9,
        'R': ['R'] * 9,
        'F': ['B'] * 9,
        'D': ['Y'] * 9,
        'L': ['O'] * 9,
        'B': ['G'] * 9,
    }

if 'results' not in st.session_state:
    st.session_state.results = []

# ── Header ─────────────────────────────────────────────────────────

st.title("Rubik's Cube Solver")
st.markdown(
    "**Improving Rubik's Cube Solving with Neural-Guided Move Ordering in IDA*** "
    "— Akash Chakraborty & Atul Chaudhuri "
    "*(Accepted: CIIR 2026, Springer LNNS)*"
)

# ── Sidebar: Mode + Controls ──────────────────────────────────────

with st.sidebar:
    st.header("Solver Settings")

    mode = st.radio(
        "Solver Mode",
        list(SOLVER_MODES.keys()),
        index=1,
        help="Choose which algorithm variant to use",
    )
    st.caption(SOLVER_MODES[mode]['desc'])

    st.divider()

    if st.button("Scramble", use_container_width=True, type='secondary'):
        st.session_state.cube = random_scramble_state()
        st.session_state.results = []
        st.rerun()

    if st.button("Reset (Solved)", use_container_width=True):
        st.session_state.cube = {
            'U': ['W'] * 9, 'R': ['R'] * 9, 'F': ['B'] * 9,
            'D': ['Y'] * 9, 'L': ['O'] * 9, 'B': ['G'] * 9,
        }
        st.session_state.results = []
        st.rerun()

    st.divider()
    st.header("Compare All Modes")
    if st.button("Solve with ALL 4 modes", use_container_width=True, type='primary'):
        cube_str = cube_state_to_string(st.session_state.cube)
        if not validate_cube_string(cube_str):
            st.error("Invalid cube — each color must appear exactly 9 times.")
        else:
            results = []
            for m_name in SOLVER_MODES:
                try:
                    sol, ms = solve_with_mode(cube_str, m_name)
                    moves = len(sol.split())
                    results.append({
                        'mode': m_name, 'moves': moves,
                        'time_ms': ms, 'solution': sol,
                    })
                except Exception as e:
                    results.append({
                        'mode': m_name, 'moves': '-',
                        'time_ms': 0, 'solution': str(e),
                    })
            st.session_state.results = results
            st.rerun()

    # Show comparison results
    if st.session_state.results:
        st.markdown("**Results:**")
        for r in st.session_state.results:
            if isinstance(r['moves'], int):
                st.markdown(
                    f"- **{r['mode']}**: {r['moves']} moves, "
                    f"{r['time_ms']:.0f} ms"
                )
            else:
                st.markdown(f"- **{r['mode']}**: timeout/error")

# ── Main: Face Input ───────────────────────────────────────────────

st.subheader("Enter Cube State")
st.caption(
    "Set each sticker color to match your physical cube. "
    "Centers are locked. Or press **Scramble** for a random cube."
)

# Show faces in 2 rows of 3
for row_start in range(0, 6, 3):
    face_cols = st.columns(3)
    for i in range(3):
        face_idx = row_start + i
        face_key, face_name, center_color, color_name = FACES[face_idx]
        with face_cols[i]:
            st.markdown(f"**{face_name}** ({color_name} center)")
            render_face_grid(
                st.session_state.cube[face_key],
                face_key,
                editable=True,
            )

# ── Solve Button ───────────────────────────────────────────────────

st.divider()

col_solve, col_info = st.columns([1, 2])

with col_solve:
    solve_clicked = st.button(
        f"Solve ({mode})",
        type='primary',
        use_container_width=True,
    )

if solve_clicked:
    cube_str = cube_state_to_string(st.session_state.cube)

    if not validate_cube_string(cube_str):
        st.error("Invalid cube configuration — each color must appear exactly 9 times.")
    else:
        with st.spinner(f"Solving with {mode}..."):
            try:
                solution, elapsed = solve_with_mode(cube_str, mode)
                moves = solution.split()

                st.success(
                    f"**{mode}**: {len(moves)} moves in {elapsed:.0f} ms"
                )
                st.code(solution, language=None)

                st.markdown("**Step by step:**")
                move_names = {
                    'U': 'Up', 'D': 'Down', 'R': 'Right',
                    'L': 'Left', 'F': 'Front', 'B': 'Back',
                }
                for j, mv in enumerate(moves):
                    face_letter = mv[0]
                    suffix = mv[1:] if len(mv) > 1 else ''
                    direction = "clockwise 90"
                    if suffix == "'":
                        direction = "counter-clockwise 90"
                    elif suffix == "2":
                        direction = "180"
                    face_full = move_names.get(face_letter, face_letter)
                    st.markdown(
                        f"{j+1}. **{mv}** — turn {face_full} face {direction}"
                    )

            except Exception as e:
                st.error(f"Could not solve: {e}")

# ── Footer ─────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Powered by Kociemba's Two-Phase Algorithm with multi-solution search, "
    "anytime mode, and neural move ordering. "
    "[GitHub](https://github.com/Akash-Chakraborty/Rubik-s-Cube-Solver)"
)
