# Vision Recognition App

A Python application for hand gesture recognition and face detection using OpenCV and MediaPipe libraries.

## Features

- **Hand Gesture Recognition**: Detects and displays hand gestures including open palm, fist, thumbs up, peace sign, and pointing.
- **Face Detection**: Detects faces in the webcam feed using OpenCV's built-in face detection.
- **Real-time Overlay**: Displays gesture names and face detection boxes in the webcam feed.

## Requirements

- macOS (tested on macOS Monterey and later)
- Python 3.8 or later
- Webcam (built-in or external)
- `uv` package manager (recommended for dependency management)

## Installation

### 1. Install uv (if not already installed)

```bash
curl -sSf https://astral.sh/uv/install.sh | bash
```

Verify the installation:

```bash
uv --version
```

### 2. Clone or download this repository

```bash
git clone <repository-url>
cd opencv-test
```

### 3. Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
uv pip install -r requirements.txt
```

## Usage

1. Ensure your virtual environment is activated
2. Run the application:

```bash
python main.py
```

3. Position your hand in the webcam view to detect gestures
4. The application will detect faces using OpenCV's built-in face detection
5. Press 'q' to quit the application

## Supported Hand Gestures

- **Open Palm**: Show all five fingers extended
- **Fist**: Close all fingers
- **Thumbs Up**: Extend only the thumb
- **Peace Sign**: Extend index and middle fingers
- **Pointing**: Extend only the index finger

## Troubleshooting

### Webcam Not Working

Ensure your webcam is connected and not being used by another application. You may need to grant camera permissions to Terminal/IDE in System Preferences > Security & Privacy > Camera.

### Performance Issues

If the application runs slowly:

- Close other applications using the webcam or CPU resources
- Reduce the webcam resolution in the code (modify the `cv2.VideoCapture` settings)

## License

MIT
