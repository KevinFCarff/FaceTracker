# Face Tracking Application

A real-time face tracking application using MediaPipe and OpenCV that allows you to track facial landmarks with interactive labeling and group management capabilities.

## Features

- **Real-time face tracking** using MediaPipe's Face Landmarker
- **Interactive labeling mode** for creating custom landmark groups
- **CSV export/import** for saving and loading landmark groups
- **Command-line group selection** to track specific predefined groups
- **Custom point tracking** for specific landmark indices
- **Window dragging** (limited platform support)
- **Hover information** showing landmark details
- **Area selection** with Shift/Ctrl modifiers for group operations

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam/camera device

### Dependencies

Install the required packages:

```bash
pip install opencv-python mediapipe
```

### Model Files

The application requires these model files in the same directory:
- `detector.tflite` - Face detection model
- `face_landmarker.task` - Face landmark detection model

## Usage

### Basic Usage

Run the application with default settings:

```bash
python track.py
```

### Command-Line Options

#### Resolution
Set camera resolution:
```bash
python track.py --resolution 1280x720
```

### Command-Line Options

#### Resolution
Set camera resolution:
```bash
python track.py --resolution 1280x720
```

#### Tracking Options
- `--track-all` - Track all landmarks
- `--points "0,1,2,3,4,5"` - Track specific landmark indices
- `--groups "Eyes,Mouth,Nose"` - Load and track predefined groups from CSV

#### Display Options
- `--label-points` - Show landmark index numbers next to points
- `--hover-info` - Show landmark index when hovering over points
- `--labeling-mode` - Enable interactive labeling mode

#### Examples
```bash
# Basic usage (tracks all landmarks)
python track.py

# High resolution
python track.py --resolution 1280x720

# Track specific groups
python track.py --groups "Eyes,Mouth"

# Custom points with labels
python track.py --points "33,34,35,36" --label-points

# Labeling mode
python track.py --labeling
```

## Interactive Controls

### Window Dragging
- **Click and drag anywhere** on the window to move it (limited platform support)

### Labeling Mode

When labeling mode is enabled (`--labeling`):

#### Point Selection
- **Left-click** on landmarks to select/deselect them
- **Shift+click** to add to current selection
- **Ctrl+click** to toggle selection
- **Drag** to select multiple points in an area

#### Group Management
- **Press 'n'** to create a new group
- **Type group name** when prompted, press Enter to confirm
- **Press 's'** to save current selection to the active group
- **Press 'c'** to clear current selection
- **Press 'd'** to delete the active group
- **Press 'e'** to export groups to CSV
- **Press 'l'** to load groups from CSV

#### Navigation
- **Press 'Tab'** to cycle through groups
- **Press 'Esc'** to exit labeling mode

### General Controls
- **Press 'h'** to toggle hover information
- **Press 'q'** or **Esc** to quit

## CSV Group Format

Groups are saved to `face_landmark_groups.csv` with the following format:

```csv
Group Name,Point Indices
Eyes,33,34,35,36,37,38,39,40,41,42,43,44,45,46
Mouth,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80
```

Each row contains:
- **Group Name**: Descriptive name for the landmark group
- **Point Indices**: Comma-separated list of landmark indices (0-467)

## Landmark Indices

The application uses MediaPipe's 468 facial landmarks (0-467). Some common landmark ranges:

- **Face outline**: 0-16
- **Eyebrows**: 17-26 (left), 27-36 (right)
- **Eyes**: 33-46 (left), 133-145 (right)
- **Nose**: 1-32
- **Mouth**: 61-87
- **Cheeks**: 123-132 (left), 220-229 (right)

## Troubleshooting

### Window Dragging Not Working
Window dragging is not supported on Windows. This is a limitation of OpenCV on Windows platforms.

### Camera Not Found
- Ensure your webcam is connected and not being used by other applications
- Try changing the camera index in the code if you have multiple cameras

### Model Files Missing
Download the required model files:
- `detector.tflite`
- `face_landmarker.task`

### Performance Issues
- Lower the resolution: `--resolution 320x240`
- Track fewer features instead of `--track-all`

## Development

The main application code is in `track.py`. Key components:

- `main()` - Application entry point and argument parsing
- `visualize()` - Landmark rendering and interaction
- `mouse_callback()` - Mouse event handling
- `save_groups_to_csv()` / `load_groups_from_csv()` - CSV persistence
- Global variables for state management

## License

This project is open source. Feel free to modify and distribute.