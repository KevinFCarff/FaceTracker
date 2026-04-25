import cv2
import mediapipe as mp
import time
import argparse
import platform
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec

# Landmark indices for specific features (based on MediaPipe Face Mesh)
LANDMARK_INDICES = {
    'eyes': [
        # Left eye
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        # Right eye  
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ],
    'eyeballs': [
        # Left iris
        468, 469, 470, 471, 472, 473,
        # Right iris
        474, 475, 476, 477, 478, 479
    ],
    'eyebrows': [
        # Left eyebrow
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        # Right eyebrow
        336, 296, 334, 293, 300, 276, 283, 282, 295, 285
    ],
    'mouth': [
        # Outer lips
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        # Inner lips  
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        # Lower lip additional
        17, 84, 181, 91, 146, 314, 405, 321, 375
    ],
    'nose': [
        # Nose tip
        4, 5, 6,
        # Nose bridge
        168,
        # Nostrils
        129, 219, 237, 241, 44, 45
    ],
    'cheeks': [
        # Left cheek (key puffing points)
        101, 205, 425,
        # Right cheek (key puffing points)
        280, 350, 411
    ],
    'face_outline': [
        # Jawline and face outline
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ],
}

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

# Global variables for window dragging
DRAGGING = False
DRAG_START_X = 0
DRAG_START_Y = 0
WINDOW_START_X = 0
WINDOW_START_Y = 0
LAST_MOVE_TIME = 0
MOVE_THROTTLE_MS = 16  # ~60 FPS throttling

# Global variables for hover functionality
MOUSE_X = 0
MOUSE_Y = 0

# Global variables for labeling mode
LABELING_MODE = False
SELECTED_POINTS = set()
CURRENT_GROUP_NAME = ""
GROUPS = {}  # Dictionary to store groups: {group_name: [point_indices]}
TEXT_INPUT_MODE = False  # Track if we're in text input mode

# Global variables for area selection
SELECTING_AREA = False
SELECTION_START_X = 0
SELECTION_START_Y = 0
SELECTION_END_X = 0
SELECTION_END_Y = 0
DRAG_THRESHOLD = 10  # Minimum pixels to consider it a drag vs click
CURRENT_MODIFIERS = 0  # Store current modifier key state

# Global variables for current frame dimensions
CURRENT_FRAME_WIDTH = 640
CURRENT_FRAME_HEIGHT = 480

# Check if window dragging is supported on this platform
WINDOW_DRAGGING_SUPPORTED = platform.system() != 'Windows'

# Check if window dragging is supported on this platform
WINDOW_DRAGGING_SUPPORTED = platform.system() != 'Windows'

def save_groups_to_csv(filename="face_landmark_groups.csv"):
    """Save the defined groups to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Group Name', 'Point Indices'])
        for group_name, points in GROUPS.items():
            writer.writerow([group_name, ','.join(map(str, sorted(points)))])
    print(f"Groups saved to {filename}")

def load_groups_from_csv(filename="face_landmark_groups.csv"):
    """Load groups from a CSV file and return as dictionary."""
    groups = {}
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    group_name = row[0].strip()
                    try:
                        points = [int(x.strip()) for x in row[1].split(',') if x.strip()]
                        groups[group_name] = points
                    except ValueError:
                        print(f"Warning: Invalid point indices for group '{group_name}': {row[1]}")
        print(f"Loaded {len(groups)} groups from {filename}")
        return groups
    except FileNotFoundError:
        print(f"Warning: Groups file {filename} not found")
        return {}
    except Exception as e:
        print(f"Error loading groups from {filename}: {e}")
        return {}

def draw_labeling_ui(image):
    """Draw the labeling mode UI elements on the image."""
    h, w = image.shape[:2]
    
    # Draw mode indicator
    mode_text = "LABELING MODE"
    if TEXT_INPUT_MODE:
        mode_text += " - TEXT INPUT"
    cv2.putText(image, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Draw instructions
    if TEXT_INPUT_MODE:
        instructions = [
            "TEXT INPUT MODE - Type group name",
            "Press Enter to save group",
            "Backspace to delete character",
            "Press _ again to exit text input"
        ]
    else:
        instructions = [
            "Click: Select single point",
            "Shift+Click: Add point to selection",
            "Ctrl+Click: Remove point from selection",
            "Drag: Select area (replace selection)",
            "Shift+Drag: Add area to selection",
            "Ctrl+Drag: Remove area from selection",
            "Press _: Enter text input mode",
            "R: Reset selection",
            "E: Export groups to CSV",
            f"Selected: {len(SELECTED_POINTS)} points"
        ]
    
    for i, instruction in enumerate(instructions):
        cv2.putText(image, instruction, (10, 70 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw current group name if any
    if CURRENT_GROUP_NAME:
        cv2.putText(image, f"Current Group: {CURRENT_GROUP_NAME}_", (10, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(image, "Type group name, then press Enter", (10, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw group list
    if GROUPS:
        cv2.putText(image, "Defined Groups:", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        group_text = ", ".join([f"{name}({len(points)})" for name, points in GROUPS.items()])
        cv2.putText(image, group_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw selection rectangle if actively selecting
    global SELECTING_AREA, SELECTION_START_X, SELECTION_START_Y, SELECTION_END_X, SELECTION_END_Y, CURRENT_MODIFIERS
    if SELECTING_AREA:
        # Choose color based on modifier keys
        shift_pressed = CURRENT_MODIFIERS & cv2.EVENT_FLAG_SHIFTKEY
        ctrl_pressed = CURRENT_MODIFIERS & cv2.EVENT_FLAG_CTRLKEY
        
        if shift_pressed:
            rect_color = (0, 255, 0)  # Green for add
            mode_text = "ADD"
        elif ctrl_pressed:
            rect_color = (0, 0, 255)  # Red for remove
            mode_text = "REMOVE"
        else:
            rect_color = (255, 0, 255)  # Magenta for replace
            mode_text = "REPLACE"
        
        cv2.rectangle(image, (SELECTION_START_X, SELECTION_START_Y), 
                     (SELECTION_END_X, SELECTION_END_Y), rect_color, 2)
        
        # Show mode text
        cv2.putText(image, mode_text, (SELECTION_START_X + 5, SELECTION_START_Y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)
        
        # Fill rectangle with semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (SELECTION_START_X, SELECTION_START_Y), 
                     (SELECTION_END_X, SELECTION_END_Y), rect_color, -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for window dragging, hover detection, and labeling."""
    global DRAGGING, DRAG_START_X, DRAG_START_Y, WINDOW_START_X, WINDOW_START_Y, LAST_MOVE_TIME, MOUSE_X, MOUSE_Y
    global SELECTED_POINTS, CURRENT_GROUP_NAME, GROUPS, SELECTING_AREA, SELECTION_START_X, SELECTION_START_Y, SELECTION_END_X, SELECTION_END_Y
    
    # Always track mouse position for hover detection
    MOUSE_X, MOUSE_Y = x, y
    CURRENT_MODIFIERS = flags  # Store current modifier key state
    
    # Handle labeling mode events
    if LABELING_MODE and DETECTION_RESULT and DETECTION_RESULT.face_landmarks:
        face_landmarks = DETECTION_RESULT.face_landmarks[0]  # Use first face
        h, w = CURRENT_FRAME_HEIGHT, CURRENT_FRAME_WIDTH  # Use actual frame dimensions
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start potential area selection or single point selection
            SELECTING_AREA = False  # Will be set to True if mouse moves
            SELECTION_START_X, SELECTION_START_Y = x, y
            SELECTION_END_X, SELECTION_END_Y = x, y
            print(f"Mouse down at ({x}, {y})")
            
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            # Left button is being held down - check if this is a drag
            distance_from_start = ((x - SELECTION_START_X) ** 2 + (y - SELECTION_START_Y) ** 2) ** 0.5
            if distance_from_start > DRAG_THRESHOLD:
                SELECTING_AREA = True
                SELECTION_END_X, SELECTION_END_Y = x, y
            
        elif event == cv2.EVENT_LBUTTONUP:
            if SELECTING_AREA:
                # Complete area selection
                SELECTION_END_X, SELECTION_END_Y = x, y
                SELECTING_AREA = False
                
                # Calculate selection rectangle bounds
                min_x = min(SELECTION_START_X, SELECTION_END_X)
                max_x = max(SELECTION_START_X, SELECTION_END_X)
                min_y = min(SELECTION_START_Y, SELECTION_END_Y)
                max_y = max(SELECTION_START_Y, SELECTION_END_Y)
                
                # Find all points within the selection rectangle
                selected_in_area = []
                for idx, landmark in enumerate(face_landmarks):
                    landmark_x = int(landmark.x * w)
                    landmark_y = int(landmark.y * h)
                    
                    if min_x <= landmark_x <= max_x and min_y <= landmark_y <= max_y:
                        selected_in_area.append(idx)
                
                if selected_in_area:
                    # Apply selection based on modifier keys
                    shift_pressed = flags & cv2.EVENT_FLAG_SHIFTKEY
                    ctrl_pressed = flags & cv2.EVENT_FLAG_CTRLKEY
                    
                    if shift_pressed:
                        # Add points to selection
                        for idx in selected_in_area:
                            SELECTED_POINTS.add(idx)
                        print(f"Added {len(selected_in_area)} points to selection (Shift+drag)")
                    elif ctrl_pressed:
                        # Remove points from selection
                        for idx in selected_in_area:
                            SELECTED_POINTS.discard(idx)
                        print(f"Removed {len(selected_in_area)} points from selection (Ctrl+drag)")
                    else:
                        # Replace selection
                        SELECTED_POINTS.clear()
                        SELECTED_POINTS.update(selected_in_area)
                        print(f"Selected {len(selected_in_area)} points (drag)")
                    
                    print(f"Selected points: {sorted(SELECTED_POINTS)}")
                else:
                    print("No points found in selection area")
            else:
                # This was a click, not a drag - do single point selection
                image_x, image_y = x, y
                print(f"Mouse click at ({x}, {y})")
                
                # Find closest landmark to click position
                min_distance = float('inf')
                closest_idx = -1
                
                for idx, landmark in enumerate(face_landmarks):
                    landmark_x = int(landmark.x * w)
                    landmark_y = int(landmark.y * h)
                    distance = ((image_x - landmark_x) ** 2 + (image_y - landmark_y) ** 2) ** 0.5
                    
                    if distance < min_distance and distance <= 25:  # Click threshold
                        min_distance = distance
                        closest_idx = idx
                
                print(f"Closest point: {closest_idx}, distance: {min_distance:.1f}")
                
                if closest_idx != -1:
                    shift_pressed = flags & cv2.EVENT_FLAG_SHIFTKEY
                    ctrl_pressed = flags & cv2.EVENT_FLAG_CTRLKEY
                    
                    if shift_pressed:
                        # Add point to selection
                        SELECTED_POINTS.add(closest_idx)
                        print(f"Added point {closest_idx} to selection (Shift+click)")
                    elif ctrl_pressed:
                        # Remove point from selection
                        SELECTED_POINTS.discard(closest_idx)
                        print(f"Removed point {closest_idx} from selection (Ctrl+click)")
                    else:
                        # Replace selection with single point
                        SELECTED_POINTS.clear()
                        SELECTED_POINTS.add(closest_idx)
                        print(f"Selected point {closest_idx} (click)")
                else:
                    print(f"No point found near click at ({x}, {y})")
        
        elif event == cv2.EVENT_RBUTTONDOWN and not SELECTING_AREA:
            # Save current selection as a group
            if SELECTED_POINTS and CURRENT_GROUP_NAME:
                GROUPS[CURRENT_GROUP_NAME] = list(SELECTED_POINTS)
                print(f"Saved group '{CURRENT_GROUP_NAME}' with {len(SELECTED_POINTS)} points")
                SELECTED_POINTS.clear()
                CURRENT_GROUP_NAME = ""
            elif SELECTED_POINTS:
                print("Please enter a group name first (use text input)")
            else:
                print("No points selected to save")
    
    # Only handle dragging events if supported and not in labeling mode
    if not WINDOW_DRAGGING_SUPPORTED or LABELING_MODE:
        return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start dragging
        DRAGGING = True
        DRAG_START_X = x
        DRAG_START_Y = y
        # Get current window position
        try:
            rect = cv2.getWindowImageRect("Face Tracking")
            if rect and len(rect) >= 4:
                WINDOW_START_X, WINDOW_START_Y = rect[0], rect[1]
            else:
                DRAGGING = False  # Disable dragging if we can't get window position
        except:
            # getWindowImageRect not supported on this platform
            DRAGGING = False
            
    elif event == cv2.EVENT_MOUSEMOVE and DRAGGING:
        # Throttle moves to prevent excessive updates
        current_time = time.time() * 1000  # milliseconds
        if current_time - LAST_MOVE_TIME < MOVE_THROTTLE_MS:
            return
            
        LAST_MOVE_TIME = current_time
        
        # Calculate new window position
        # x, y are window-relative coordinates, so we need to calculate the delta
        delta_x = x - DRAG_START_X
        delta_y = y - DRAG_START_Y
        new_x = WINDOW_START_X + delta_x
        new_y = WINDOW_START_Y + delta_y
        
        # Basic bounds checking (prevent window from going completely off-screen)
        screen_width, screen_height = 1920, 1080  # Default fallback
        try:
            # Try to get screen size, but this might not work on all systems
            pass  # For now, skip screen size detection
        except:
            pass
            
        # Ensure window doesn't go too far off-screen (allow some off-screen for usability)
        if new_x > -100 and new_y > -100:  # Allow window to be mostly off-screen
            try:
                cv2.moveWindow("Face Tracking", new_x, new_y)
            except:
                # moveWindow may not be supported on all platforms
                pass
                
    elif event == cv2.EVENT_LBUTTONUP:
        # Stop dragging
        DRAGGING = False

def visualize(image, detection_result, track_features, custom_points=None, label_points=False, hover_info=False, mouse_pos=None, labeling_mode=False):
    """Draws selected facial landmarks on the input image and return it."""
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            # Filter landmarks based on selected features
            selected_indices = set()
            
            if ('custom' in track_features or 'groups' in track_features) and custom_points:
                # Use custom points (either specified directly or loaded from groups)
                selected_indices = set(custom_points)
            else:
                # Use predefined feature groups
                for feature in track_features:
                    if feature in LANDMARK_INDICES:
                        selected_indices.update(LANDMARK_INDICES[feature])
            
            if not selected_indices:
                # If no specific features, draw all landmarks
                selected_landmarks = face_landmarks
                indices_to_draw = list(range(len(face_landmarks)))
            else:
                # Create a filtered list of landmarks
                selected_landmarks = [face_landmarks[i] for i in sorted(selected_indices) if i < len(face_landmarks)]
                indices_to_draw = sorted(selected_indices)
            
            # Draw the selected landmarks
            if labeling_mode:
                # In labeling mode, draw all landmarks with different colors for selected ones
                h, w, _ = image.shape
                for idx, landmark in enumerate(face_landmarks):
                    landmark_x = int(landmark.x * w)
                    landmark_y = int(landmark.y * h)
                    
                    # Choose color based on selection status
                    if idx in SELECTED_POINTS:
                        color = (0, 255, 255)  # Yellow for selected
                        radius = 4
                        # Always show labels for selected points
                        cv2.putText(image, str(idx), (landmark_x + 5, landmark_y - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        color = (0, 255, 0)    # Green for unselected
                        radius = 2
                        # Only show labels on hover if hover_info is enabled
                        if hover_info and mouse_pos:
                            mouse_x, mouse_y = mouse_pos
                            distance = ((mouse_x - landmark_x) ** 2 + (mouse_y - landmark_y) ** 2) ** 0.5
                            if distance <= 10:
                                cv2.putText(image, str(idx), (landmark_x + 5, landmark_y - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    cv2.circle(image, (landmark_x, landmark_y), radius, color, -1)
            else:
                # Normal drawing mode
                drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=selected_landmarks,
                    connections=None,  # Draw points only
                    landmark_drawing_spec=DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
                
                # Label points with indices if requested
                if label_points:
                    h, w, _ = image.shape
                    for idx, landmark in zip(indices_to_draw, selected_landmarks):
                        # Convert normalized coordinates to pixel coordinates
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        # Draw the index number
                        cv2.putText(image, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Show hover info if enabled (only in non-labeling mode)
            if hover_info and mouse_pos:
                mouse_x, mouse_y = mouse_pos
                h, w, _ = image.shape
                hover_threshold = 10  # pixels
                
                if labeling_mode:
                    # In labeling mode, only show hover for unselected points
                    for idx, landmark in enumerate(face_landmarks):
                        if idx in SELECTED_POINTS:
                            continue  # Skip selected points, they already show labels
                        
                        landmark_x = int(landmark.x * w)
                        landmark_y = int(landmark.y * h)
                        
                        distance = ((mouse_x - landmark_x) ** 2 + (mouse_y - landmark_y) ** 2) ** 0.5
                        if distance <= hover_threshold:
                            cv2.putText(image, str(idx), (landmark_x + 5, landmark_y - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.circle(image, (landmark_x, landmark_y), 6, (255, 255, 0), 2)
                            print(f"Hovering over unselected point {idx} at ({landmark_x}, {landmark_y})")
                            break
                else:
                    # Normal hover mode
                    for idx, landmark in zip(indices_to_draw, selected_landmarks):
                        landmark_x = int(landmark.x * w)
                        landmark_y = int(landmark.y * h)
                        
                        distance = ((mouse_x - landmark_x) ** 2 + (mouse_y - landmark_y) ** 2) ** 0.5
                        if distance <= hover_threshold:
                            cv2.putText(image, f"Point {idx}", (mouse_x + 15, mouse_y - 15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                            cv2.circle(image, (landmark_x, landmark_y), 8, (255, 255, 0), 2)
                            print(f"Hovering over point {idx} at ({landmark_x}, {landmark_y}), mouse at ({mouse_x}, {mouse_y})")
                            break
    return image

def save_result(result: vision.FaceLandmarkerResult, unused_output_image: mp.Image, timestamp_ms: int):
    global FPS, COUNTER, START_TIME, DETECTION_RESULT

    # Calculate the FPS
    if COUNTER % 10 == 0:
        FPS = 10 / (time.time() - START_TIME)
        START_TIME = time.time()

    DETECTION_RESULT = result
    COUNTER += 1

def main():
    # Declare global variables that will be modified
    global LABELING_MODE, CURRENT_GROUP_NAME, SELECTED_POINTS, GROUPS, TEXT_INPUT_MODE
    
    # Initialize local variables
    selected_groups = None
    
    parser = argparse.ArgumentParser(description='Real-time face tracking with MediaPipe')
    parser.add_argument('-r', '--resolution', type=str, default='640x480', 
                        help='Camera resolution in WIDTHxHEIGHT format (e.g., 640x480, 1280x720)')
    parser.add_argument('-a', '--track-all', action='store_true', help='Track all landmarks (default if no specific features selected)')
    parser.add_argument('-l', '--label-points', action='store_true', help='Show landmark index numbers next to points')
    parser.add_argument('-p', '--points', type=str, help='Comma-separated list of specific landmark indices to track (e.g., "1,2,3,4,5")')
    parser.add_argument('-i', '--hover-info', action='store_true', help='Show landmark index when hovering over points with mouse')
    parser.add_argument('-g', '--labeling-mode', action='store_true', help='Enable labeling mode for creating custom point groups')
    parser.add_argument('-G', '--groups', type=str, help='Load and track specific groups from CSV file (comma-separated, e.g., "Eyes,Mouth,Nose")')
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        print("Invalid resolution format. Use WIDTHxHEIGHT (e.g., 640x480)")
        return
    
    # Determine which features to track
    track_features = []
    
    # Handle custom points
    custom_points = None
    if args.points:
        try:
            custom_points = [int(x.strip()) for x in args.points.split(',')]
            track_features = ['custom']  # Special marker for custom points
        except ValueError:
            print(f"Invalid points format: {args.points}. Use comma-separated integers.")
            return
    
    # Handle groups from CSV
    selected_groups = None
    if args.groups:
        loaded_groups = load_groups_from_csv()
        if loaded_groups:
            try:
                group_names = [name.strip() for name in args.groups.split(',')]
                selected_groups = []
                all_points = set()
                
                for group_name in group_names:
                    if group_name in loaded_groups:
                        selected_groups.append(group_name)
                        all_points.update(loaded_groups[group_name])
                    else:
                        print(f"Warning: Group '{group_name}' not found in CSV file")
                
                if all_points:
                    custom_points = sorted(list(all_points))
                    track_features = ['groups']  # Special marker for loaded groups
                    print(f"Loaded groups: {', '.join(selected_groups)} ({len(custom_points)} total points)")
                else:
                    print("No valid groups found")
            except Exception as e:
                print(f"Error processing groups: {e}")
        else:
            print("Could not load groups from CSV file")
    
    if args.track_all or (not track_features and not custom_points):
        track_features = ['all']  # Special case for all landmarks
    
    label_points = args.label_points
    hover_info = args.hover_info
    labeling_mode = args.labeling_mode
    
    # Set global labeling mode
    global LABELING_MODE
    LABELING_MODE = labeling_mode
    
    # Start webcam with specified resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Check actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested resolution: {width}x{height}")
    print(f"Actual camera resolution: {actual_width}x{actual_height}")
    
    if actual_width != width or actual_height != height:
        print("Warning: Camera does not support requested resolution, using closest available")
        width, height = actual_width, actual_height
    
    # Initialize the face landmarker model
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=save_result
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    print(f"Starting face tracking at {width}x{height} resolution")
    if selected_groups:
        print(f"Tracking groups: {', '.join(selected_groups)} ({len(custom_points)} points)")
    elif custom_points:
        print(f"Tracking custom points: {len(custom_points)} points")
    else:
        print(f"Tracking features: {', '.join(track_features)}")
    if WINDOW_DRAGGING_SUPPORTED:
        print("Window dragging: Enabled (click and drag anywhere on the window)")
    else:
        print("Window dragging: Disabled (not supported on Windows)")
    if hover_info:
        print("Hover info: Enabled (hover over points to see indices)")
    if labeling_mode:
        print("Labeling mode: Enabled (create custom point groups)")
        print("Controls: Left click to select/deselect points, R to reset, E to export CSV")
    
    # Set up mouse callback for window dragging and hover detection
    cv2.namedWindow("Face Tracking", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Face Tracking", mouse_callback)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect (optional)
        frame = cv2.flip(frame, 1)

        # Convert to RGB (MediaPipe requires this)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run face landmarker
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Update current frame dimensions for mouse callback
        global CURRENT_FRAME_WIDTH, CURRENT_FRAME_HEIGHT
        CURRENT_FRAME_HEIGHT, CURRENT_FRAME_WIDTH = frame.shape[:2]

        # Draw landmarks
        if DETECTION_RESULT:
            frame = visualize(frame, DETECTION_RESULT, track_features, custom_points, label_points, hover_info, (MOUSE_X, MOUSE_Y), labeling_mode)

        # Draw labeling UI if in labeling mode
        if labeling_mode:
            draw_labeling_ui(frame)

        # Show FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (24, 50)
        cv2.putText(frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    1, (0, 0, 0), 1, cv2.LINE_AA)

        # Show result
        cv2.imshow("Face Tracking", frame)

        # Exit on 'q' or ESC
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
            
        # Handle labeling mode keyboard shortcuts and text input
        if labeling_mode:
            # Toggle text input mode with underscore
            if key & 0xFF == ord('_'):
                TEXT_INPUT_MODE = not TEXT_INPUT_MODE
                if TEXT_INPUT_MODE:
                    print("Entered text input mode - type group name, press Enter to save")
                else:
                    print("Exited text input mode")
            
            elif TEXT_INPUT_MODE:
                # Text input mode - handle typing
                if key != -1 and key != 0:  # Some key was pressed
                    char = chr(key & 0xFF)
                    if char.isalnum() or char in ['_', '-', ' ', 'e', 'r', 'E', 'R']:
                        CURRENT_GROUP_NAME += char
                        print(f"Group name: {CURRENT_GROUP_NAME}")
                    elif key & 0xFF == 8:  # Backspace
                        CURRENT_GROUP_NAME = CURRENT_GROUP_NAME[:-1]
                        print(f"Group name: {CURRENT_GROUP_NAME}")
                    elif key & 0xFF == 13:  # Enter
                        if SELECTED_POINTS and CURRENT_GROUP_NAME:
                            GROUPS[CURRENT_GROUP_NAME] = list(SELECTED_POINTS)
                            print(f"Saved group '{CURRENT_GROUP_NAME}' with {len(SELECTED_POINTS)} points")
                            SELECTED_POINTS.clear()
                            CURRENT_GROUP_NAME = ""
                            TEXT_INPUT_MODE = False  # Exit text input mode after saving
                        else:
                            print("Cannot save: no points selected or no group name")
            else:
                # Command mode - handle shortcuts
                if key & 0xFF == ord('r') or key & 0xFF == ord('R'):
                    # Reset selection
                    SELECTED_POINTS.clear()
                    CURRENT_GROUP_NAME = ""
                    TEXT_INPUT_MODE = False
                    print("Selection reset")
                elif key & 0xFF == ord('e') or key & 0xFF == ord('E'):
                    # Export to CSV
                    if GROUPS:
                        save_groups_to_csv()
                    else:
                        print("No groups to export")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()