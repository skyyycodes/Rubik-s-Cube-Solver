import cv2 as cv
import numpy as np
import color_detect
import maker
import MYcubePreview
import Info
from kociemba import kociemba
import subprocess
import sys

# Manual input mode variables
manual_mode = True  # Start in manual mode by default
current_face = 'front'  # Current face being edited in manual mode
current_position = 0    # Current position (0-8) on the face
face_order = ['up', 'right', 'front', 'down', 'left', 'back']
face_names = {'up': 'Upper (White)', 'right': 'Right (Red)', 'front': 'Front (Blue)',
              'down': 'Down (Yellow)', 'left': 'Left (Orange)', 'back': 'Back (Green)'}
# Mapping from cube keys to preview side names
preview_side_names = {'up': 'upper', 'right': 'right', 'front': 'front',
                      'down': 'down', 'left': 'left', 'back': 'back'}

# vid object to capture frames from the camera and process them in code
vid = cv.VideoCapture(0)
# giving height and widht to vis frame
vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

cube = {'up' : ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white'],
        'right': ['red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red'],
        'front': ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue'],
        'down': ['yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow'],
        'left': ['orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange'],
        'back': ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green']}


# empty screen for screen1 and screen2
blank = np.zeros((950, 600, 3) , dtype='uint8')
blank2 = np.zeros((230, 1280, 3) , dtype='uint8')

def draw_info_panel(manual_mode, current_face, current_position):
    """Redraws the info panel based on current mode"""
    blank2[:] = 0  # Clear the panel

    if not manual_mode:
        # Camera mode instructions
        cv.putText(blank2, 'W-upper', (0, 25), cv.FONT_ITALIC , 1, (255, 255, 255))
        cv.putText(blank2, 'Z-down', (0, 65), cv.FONT_ITALIC , 1, (0, 255, 255))
        cv.putText(blank2, 'A-left', (0, 105), cv.FONT_ITALIC , 1, (0, 140, 255))
        cv.putText(blank2, 'D-right', (0, 145), cv.FONT_ITALIC , 1, (0, 0, 255))
        cv.putText(blank2, 'F-back', (0, 185), cv.FONT_ITALIC , 1, (0, 255, 0))
        cv.putText(blank2, 'S-front', (0, 225), cv.FONT_ITALIC , 1, (255, 0, 0))
        cv.putText(blank2, 'Enter-Compute Solution', (400, 30), cv.FONT_ITALIC , 1.2, (255, 255, 255))
        cv.putText(blank2, 'M - Manual Mode | 3 - 3D Mode', (400, 70), cv.FONT_ITALIC , 1, (0, 255, 255))
    else:
        # Manual mode instructions - reorganized layout
        cv.putText(blank2, 'MANUAL MODE', (10, 30), cv.FONT_ITALIC, 1, (0, 255, 255))
        cv.putText(blank2, f'{face_names[current_face]}', (10, 60), cv.FONT_ITALIC, 0.6, (255, 255, 255))
        cv.putText(blank2, f'Pos: {current_position + 1}/9', (10, 85), cv.FONT_ITALIC, 0.5, (200, 200, 200))

        # Color keys - compact
        cv.putText(blank2, '1:W 2:R 3:B 4:Y 5:O 6:G', (10, 115), cv.FONT_ITALIC, 0.5, (255, 255, 255))

        # Navigation - compact
        cv.putText(blank2, 'Tab:Next `:Prev R:Reset', (10, 140), cv.FONT_ITALIC, 0.45, (180, 180, 180))

        # Right side - actions
        cv.putText(blank2, 'Enter: Solve | M: Camera | 3: 3D Mode', (10, 200), cv.FONT_ITALIC, 0.6, (150, 255, 150))

# Initial draw
draw_info_panel(manual_mode, current_face, current_position)

# side pannel for preview
MYcubePreview.default(blank)
AnsIndex = -1
text = 'deafult'

def draw_manual_input_grid(frame, cube, current_face, current_position):
    """Draw the manual input grid showing current face colors"""
    face_colors = cube[current_face]

    # Draw 3x3 grid at same position as camera detection grid
    start_x, start_y = 800, 200
    cell_size = 100

    for i in range(9):
        row = i // 3
        col = i % 3
        x1 = start_x + col * cell_size
        y1 = start_y + row * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size

        # Get color for this cell
        color = color_detect.fun2(face_colors[i])
        cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness=cv.FILLED)

        # Show position number in corner
        cv.putText(frame, str(i+1), (x1+5, y1+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Highlight current position (white border)
        if i == current_position:
            cv.rectangle(frame, (x1+5, y1+5), (x2-5, y2-5), (255, 255, 255), thickness=3)
        # Mark center as locked (X pattern)
        if i == 4:
            cv.line(frame, (x1+30, y1+30), (x2-30, y2-30), (0, 0, 0), 2)
            cv.line(frame, (x1+30, y2-30), (x2-30, y1+30), (0, 0, 0), 2)

    # Draw grid lines
    for i in range(4):
        cv.line(frame, (start_x + i * cell_size, start_y),
                (start_x + i * cell_size, start_y + 3 * cell_size), (0, 0, 0), 3)
        cv.line(frame, (start_x, start_y + i * cell_size),
                (start_x + 3 * cell_size, start_y + i * cell_size), (0, 0, 0), 3)

    # Show face name above grid
    cv.putText(frame, face_names[current_face], (start_x, start_y - 20),
               cv.FONT_ITALIC, 0.8, (255, 255, 255), 2)

    # Show orientation hint below grid
    cv.putText(frame, "Hold cube: White=Top, Blue=Front", (start_x - 50, start_y + 330),
               cv.FONT_ITALIC, 0.6, (200, 200, 200), 1)

while True:

    # reads one frame from the video source
    # ret -> boolean variable indicating whether the frame was successfully read
    # frame -> NumPy array representing the image data of the captured frame
    ret, frame= vid.read()

    # horizontal flipping the frame - 1
    frame = cv.flip(frame, 1)

    # -> fx and fy are scaling factors

    # interpolation = cv.INTER_CUBIC -> 
    # Cubic interpolation considers a larger neighborhood of pixels when estimating the 
    # value of a new pixel, resulting in smoother and more visually appealing resized images
    frame = cv.resize(frame, (1280, 720), fx=0, fy=0, interpolation = cv.INTER_CUBIC)

    # HSV (Hue-Saturation-Value) frame from BGR
    # why ? -> for color detection, for easily
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    height, width = frame.shape[:2] 
    # print(height, width)

    # Solution(frame)

    # corrdinates on window, for accessing color
    tr = [120, 120]
    tm = [360, 120]
    tl = [600, 120]
    mr = [120, 360]
    mm = [360, 360]
    ml = [600, 360]
    br = [120, 600]
    bm = [360, 600]
    bl = [600, 600]

    # img center radius color thickness
    cv.circle(frame, tr, 15, (255, 255, 255), thickness=cv.FILLED)
    cv.circle(frame, tm, 15, (255, 255, 255), thickness=cv.FILLED)
    cv.circle(frame, tl, 15, (255, 255, 255), thickness=cv.FILLED)
    cv.circle(frame, mr, 15, (255, 255, 255), thickness=cv.FILLED)
    cv.circle(frame, mm, 15, (255, 255, 255), thickness=cv.FILLED)
    cv.circle(frame, ml, 15, (255, 255, 255), thickness=cv.FILLED)
    cv.circle(frame, br, 15, (255, 255, 255), thickness=cv.FILLED)
    cv.circle(frame, bm, 15, (255, 255, 255), thickness=cv.FILLED)
    cv.circle(frame, bl, 15, (255, 255, 255), thickness=cv.FILLED)


    # detecting the colors from hsv frame (returns name -> red, blue, white)
    top_left_color = color_detect.fun(hsv_frame[120, 120])
    middle_left_color = color_detect.fun(hsv_frame[360, 120])
    bottom_left_color = color_detect.fun(hsv_frame[600, 120])
    top_middle_color = color_detect.fun(hsv_frame[120, 360])
    middle_middle_color = color_detect.fun(hsv_frame[360, 360])
    bottom_middle_color = color_detect.fun(hsv_frame[600, 360])
    top_right_color = color_detect.fun(hsv_frame[120, 600])
    middle_right_color = color_detect.fun(hsv_frame[360, 600])
    bottom_right_color = color_detect.fun(hsv_frame[600, 600])

    # fetching current frame colors, and making cube
    curr = [top_left_color, top_middle_color, top_right_color,
            middle_left_color, middle_middle_color, middle_right_color,
            bottom_left_color, bottom_middle_color, bottom_right_color]

    myCube = ""

    # The & 0xFF operation is applied to extract the least significant 8 bits of the result.
    # This is often done to ensure compatibility with ASCII values of keyboard keys.

    # waitkey(3) stops the execuation for 3ms and wait for any key event
    # key stores the accsi value
    key = cv.waitKey(3) & 0xFF


    # ord() -> gives assci value

    # Toggle manual mode with 'M' key
    if(key == ord('m') or key == ord('M')):
        manual_mode = not manual_mode
        draw_info_panel(manual_mode, current_face, current_position)

    # Launch 3D mode with '3' key
    if(key == ord('3')):
        print("\nLaunching 3D Cube Input Mode...")
        print("Close the 3D window to return to this interface.\n")
        subprocess.run([sys.executable, 'cube3d.py'])

    if manual_mode:
        # Draw manual input grid instead of camera detection
        draw_manual_input_grid(frame, cube, current_face, current_position)

        # Color input keys (1-6)
        color_map = {
            ord('1'): 'white',
            ord('2'): 'red',
            ord('3'): 'blue',
            ord('4'): 'yellow',
            ord('5'): 'orange',
            ord('6'): 'green'
        }

        if key in color_map:
            # Position 4 is the center - don't allow changing it
            if current_position != 4:
                cube[current_face][current_position] = color_map[key]
                # Update preview
                MYcubePreview.CurrToMyCubePrev(cube[current_face], preview_side_names[current_face], blank)
            # Move to next position (skip center)
            current_position = (current_position + 1) % 9
            if current_position == 4:
                current_position = 5  # Skip center
            draw_info_panel(manual_mode, current_face, current_position)

        # Navigation with arrow keys (skip center position 4)
        # Left arrow
        if key == 81 or key == 2:
            current_position = (current_position - 1) % 9
            if current_position == 4:
                current_position = 3
            draw_info_panel(manual_mode, current_face, current_position)
        # Right arrow
        if key == 83 or key == 3:
            current_position = (current_position + 1) % 9
            if current_position == 4:
                current_position = 5
            draw_info_panel(manual_mode, current_face, current_position)
        # Up arrow
        if key == 82 or key == 0:
            current_position = (current_position - 3) % 9
            if current_position == 4:
                current_position = 1
            draw_info_panel(manual_mode, current_face, current_position)
        # Down arrow
        if key == 84 or key == 1:
            current_position = (current_position + 3) % 9
            if current_position == 4:
                current_position = 7
            draw_info_panel(manual_mode, current_face, current_position)

        # Tab to go to next face
        if key == 9:  # Tab key
            face_idx = face_order.index(current_face)
            current_face = face_order[(face_idx + 1) % 6]
            current_position = 0
            draw_info_panel(manual_mode, current_face, current_position)

        # Backtick (`) to go to previous face (since Shift+Tab is hard to detect)
        if key == ord('`'):
            face_idx = face_order.index(current_face)
            current_face = face_order[(face_idx - 1) % 6]
            current_position = 0
            draw_info_panel(manual_mode, current_face, current_position)

        # R to reset current face to default colors
        if key == ord('r') or key == ord('R'):
            default_colors = {'up': 'white', 'right': 'red', 'front': 'blue',
                              'down': 'yellow', 'left': 'orange', 'back': 'green'}
            cube[current_face] = [default_colors[current_face]] * 9
            MYcubePreview.CurrToMyCubePrev(cube[current_face], preview_side_names[current_face], blank)
            current_position = 0
            draw_info_panel(manual_mode, current_face, current_position)

    else:
        # Camera mode - original functionality
        # live detection window for colors
        # img, pt1, pt2, color, thickness           # this represents proper view of colors, like perfect red
        cv.rectangle(frame, (800, 200), (900, 300), color_detect.fun2(top_left_color), thickness=cv.FILLED)
        cv.rectangle(frame, (900, 200), (1000, 300), color_detect.fun2(top_middle_color), thickness=cv.FILLED)
        cv.rectangle(frame, (1000, 200), (1100, 300), color_detect.fun2(top_right_color), thickness=cv.FILLED)
        cv.rectangle(frame, (800, 300), (900, 400), color_detect.fun2(middle_left_color), thickness=cv.FILLED)
        cv.rectangle(frame, (900, 300), (1000, 400), color_detect.fun2(middle_middle_color), thickness=cv.FILLED)
        cv.rectangle(frame, (1000, 300), (1100, 400), color_detect.fun2(middle_right_color), thickness=cv.FILLED)
        cv.rectangle(frame, (800, 400), (900, 500), color_detect.fun2(bottom_left_color), thickness=cv.FILLED)
        cv.rectangle(frame, (900, 400), (1000, 500), color_detect.fun2(bottom_middle_color), thickness=cv.FILLED)
        cv.rectangle(frame, (1000, 400), (1100, 500), color_detect.fun2(bottom_right_color), thickness=cv.FILLED)

        # white lines for displaying
        cv.line(frame, (800, 200), (800, 500), (0, 0, 0), 3)
        cv.line(frame, (900, 200), (900, 500), (0, 0, 0), 3)
        cv.line(frame, (1000, 200), (1000, 500), (0, 0, 0), 3)
        cv.line(frame, (1100, 200), (1100, 500), (0, 0, 0), 3)

        cv.line(frame, (800, 200), (1100, 200), (0, 0, 0), 3)
        cv.line(frame, (800, 300), (1100, 300), (0, 0, 0), 3)
        cv.line(frame, (800, 400), (1100, 400), (0, 0, 0), 3)
        cv.line(frame, (800, 500), (1100, 500), (0, 0, 0), 3)

        if(key == ord('b')):
            if middle_left_color == "white":
                cube = maker.Curr2Cube(cube, curr, 'upper')
                # blank is right pannel
                MYcubePreview.CurrToMyCubePrev(curr, 'upper', blank)



        if(middle_middle_color == "white"):
            # updates the cube's current side, wrt to key press
            cube = maker.Curr2Cube(cube, curr, 'upper')
            # blank is right pannel
            MYcubePreview.CurrToMyCubePrev(curr, 'upper', blank)

        if(key == ord('d')):
            cube = maker.Curr2Cube(cube, curr, 'right')
            MYcubePreview.CurrToMyCubePrev(curr, 'right', blank)

        if(key == ord('s')):
            cube = maker.Curr2Cube(cube, curr, 'front')
            MYcubePreview.CurrToMyCubePrev(curr, 'front', blank)

        if(key == ord('z')):
            cube = maker.Curr2Cube(cube, curr, 'down')
            MYcubePreview.CurrToMyCubePrev(curr, 'down', blank)

        if(key == ord('a')):
            cube = maker.Curr2Cube(cube, curr, 'left')
            MYcubePreview.CurrToMyCubePrev(curr, 'left', blank)

        if(key == ord('f')):
            cube = maker.Curr2Cube(cube, curr, 'back')
            MYcubePreview.CurrToMyCubePrev(curr, 'back', blank) 

    # The carriage return character typically moves the cursor 
    # to the beginning of the line without advancing to the next line.
    if(key == ord('\r')):
        myCube = maker.cubeToString(cube, myCube)
        print(f"Cube string: {myCube}")
        print(f"Color counts - U:{myCube.count('U')} R:{myCube.count('R')} F:{myCube.count('F')} D:{myCube.count('D')} L:{myCube.count('L')} B:{myCube.count('B')}")

        # checking valid cube, like 9 red, 9 blue, 9 white ...
        if(maker.check(myCube) == False):
            print('invalid input - color count mismatch')
            text = 'invalid cube'
        else:
            try:
                ans = kociemba.solve(myCube)
                text = ans
                ans = ans.split()
                for i in range(len(ans)):
                    print(i+1, ans[i])
            except ValueError as e:
                print(f'Invalid cube arrangement: {e}')
                text = 'invalid cube'
                ans = []
        

        if(text == 'invalid cube'):
            # printing invalid cube on screen
            Info.outputIV(text, blank2)
        else:
            # else printing correct seq of answer
            Info.outputAns(text, blank2)
            n = len(ans)
        
    # on pressing l window shows the next seq in the window
    if(key == ord('l')):
        AnsIndex = AnsIndex + 1
        
    if(text != 'invalid cube' and text != 'deafult'):
        Info.output(text, frame, AnsIndex)


    cv.imshow("frame", frame)
    cv.imshow("MyCube", blank)
    cv.imshow("Info", blank2)

    if(key == 27):
        break


vid.release()
cv.destroyAllWindows()
