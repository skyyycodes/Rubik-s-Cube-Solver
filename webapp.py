import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import color_detect
import maker

try:
    # Repository layout: ./kociemba/kociemba exposes the enhanced solver.
    from kociemba.kociemba import solve as kociemba_solve
except ImportError:
    # Fallback for pip package that exports solve at top-level.
    from kociemba import solve as kociemba_solve


def solve_cube_compat(cube_string):
    """Solve with enhanced args when available; fall back to basic API."""
    try:
        return kociemba_solve(
            cube_string,
            max_phase1_solutions=5,
            time_budget_sec=0.5,
        )
    except TypeError:
        return kociemba_solve(cube_string)

st.set_page_config(
    page_title="Rubik's Cube Solver",
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 Rubik's Cube Solver")
st.markdown("Capture each face of your cube using the webcam, then click **Solve** to get the solution!")

# Face configuration
FACES = {
    'up': {'name': 'Upper (White center)', 'color': 'white'},
    'right': {'name': 'Right (Red center)', 'color': 'red'},
    'front': {'name': 'Front (Blue center)', 'color': 'blue'},
    'down': {'name': 'Down (Yellow center)', 'color': 'yellow'},
    'left': {'name': 'Left (Orange center)', 'color': 'orange'},
    'back': {'name': 'Back (Green center)', 'color': 'green'}
}

FACE_ORDER = ['up', 'right', 'front', 'down', 'left', 'back']

# Color to RGB mapping for display
COLOR_RGB = {
    'white': '#FFFFFF',
    'red': '#FF0000',
    'blue': '#0000FF',
    'yellow': '#FFFF00',
    'orange': '#FF8C00',
    'green': '#00FF00'
}

def extract_colors_from_image(image, face_key):
    """Extract 9 colors from the captured image."""
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    img_bgr = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)

    # Resize to standard size
    img_bgr = cv.resize(img_bgr, (720, 720))

    # Convert to HSV
    hsv_frame = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    # Sample points (3x3 grid)
    h, w = img_bgr.shape[:2]
    positions = [
        (h//6, w//6), (h//6, w//2), (h//6, 5*w//6),
        (h//2, w//6), (h//2, w//2), (h//2, 5*w//6),
        (5*h//6, w//6), (5*h//6, w//2), (5*h//6, 5*w//6)
    ]

    colors = []
    for y, x in positions:
        hsv_value = hsv_frame[y, x]
        color = color_detect.fun(hsv_value)
        colors.append(color)

    # Force center to be the expected color for this face
    colors[4] = FACES[face_key]['color']

    return colors

def draw_face_preview(colors):
    """Create HTML for a 3x3 color grid preview."""
    html = '<div style="display: grid; grid-template-columns: repeat(3, 40px); gap: 2px; margin: 10px 0;">'
    for color in colors:
        rgb = COLOR_RGB.get(color, '#808080')
        html += f'<div style="width: 40px; height: 40px; background-color: {rgb}; border: 1px solid #333;"></div>'
    html += '</div>'
    return html

# Initialize session state
if 'cube' not in st.session_state:
    st.session_state.cube = {
        'up': ['white'] * 9,
        'right': ['red'] * 9,
        'front': ['blue'] * 9,
        'down': ['yellow'] * 9,
        'left': ['orange'] * 9,
        'back': ['green'] * 9
    }

if 'captured_faces' not in st.session_state:
    st.session_state.captured_faces = set()

# Instructions
with st.expander("📖 Instructions", expanded=True):
    st.markdown("""
    **How to use:**
    1. Hold your Rubik's cube with **White center facing UP** and **Blue center facing YOU**
    2. Capture each face using the webcam below
    3. Verify the detected colors match your cube
    4. Click **Solve** to get the solution

    **Face order:** Upper → Right → Front → Down → Left → Back
    """)

# Create tabs for each face
tabs = st.tabs([FACES[f]['name'] for f in FACE_ORDER])

for i, face_key in enumerate(FACE_ORDER):
    with tabs[i]:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"📷 Capture {FACES[face_key]['name']}")
            camera_input = st.camera_input(
                f"Take a photo of the {FACES[face_key]['name']} face",
                key=f"camera_{face_key}"
            )

            if camera_input is not None:
                image = Image.open(camera_input)
                colors = extract_colors_from_image(image, face_key)
                st.session_state.cube[face_key] = colors
                st.session_state.captured_faces.add(face_key)

        with col2:
            st.subheader("Detected Colors")
            colors = st.session_state.cube[face_key]
            st.markdown(draw_face_preview(colors), unsafe_allow_html=True)

            # Manual color correction
            st.markdown("**Manual correction:**")
            color_options = list(COLOR_RGB.keys())

            cols = st.columns(3)
            for j in range(9):
                if j != 4:  # Skip center (locked)
                    with cols[j % 3]:
                        new_color = st.selectbox(
                            f"Pos {j+1}",
                            color_options,
                            index=color_options.index(colors[j]),
                            key=f"color_{face_key}_{j}"
                        )
                        st.session_state.cube[face_key][j] = new_color

# Solve section
st.markdown("---")
st.subheader("🧩 Solve")

# Show cube preview
st.markdown("**Current cube state:**")
preview_cols = st.columns(6)
for i, face_key in enumerate(FACE_ORDER):
    with preview_cols[i]:
        st.caption(FACES[face_key]['name'].split(' ')[0])
        st.markdown(draw_face_preview(st.session_state.cube[face_key]), unsafe_allow_html=True)
        if face_key in st.session_state.captured_faces:
            st.success("✓ Captured")

# Solve button
if st.button("🔮 Solve Cube", type="primary", use_container_width=True):
    # Build cube string
    myCube = maker.cubeToString(st.session_state.cube, "")

    st.info(f"Cube string: `{myCube}`")

    # Validate
    if not maker.check(myCube):
        st.error("❌ Invalid cube configuration! Each color must appear exactly 9 times.")
        counts = {c: myCube.count(c) for c in 'URFDLB'}
        st.write("Color counts:", counts)
    else:
        try:
            with st.spinner("Solving..."):
                solution = solve_cube_compat(myCube)

            st.success("✅ Solution found!")
            st.markdown(f"### Solution: `{solution}`")

            # Break down moves
            moves = solution.split()
            st.markdown(f"**Total moves:** {len(moves)}")

            st.markdown("**Step by step:**")
            move_cols = st.columns(min(len(moves), 10))
            for j, move in enumerate(moves):
                with move_cols[j % 10]:
                    st.markdown(f"**{j+1}.** {move}")

        except Exception as e:
            st.error(f"❌ Could not solve: {e}")
            st.info("Make sure the cube configuration is valid and solvable.")

# Reset button
if st.button("🔄 Reset"):
    st.session_state.cube = {
        'up': ['white'] * 9,
        'right': ['red'] * 9,
        'front': ['blue'] * 9,
        'down': ['yellow'] * 9,
        'left': ['orange'] * 9,
        'back': ['green'] * 9
    }
    st.session_state.captured_faces = set()
    st.rerun()

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Powered by Kociemba's two-phase algorithm")
