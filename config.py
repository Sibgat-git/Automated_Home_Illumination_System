"""
Configuration settings for the person detection system.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

# Model settings
MODEL_PATH = MODELS_DIR / "efficientdet_lite0.tflite"
LABELS_PATH = MODELS_DIR / "coco_labels.txt"

# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for person detection
NUM_THREADS = 4  # Number of CPU threads for inference (Pi 4 has 4 cores)

# Camera settings
CAMERA_INDEX = 0  # 0 for USB camera, or use picamera2 for Pi Camera
CAMERA_RESOLUTION = (640, 480)  # Width x Height
CAMERA_FPS = 30

# Display settings
SHOW_PREVIEW = True  # Show live preview window
DRAW_BOUNDING_BOXES = True  # Draw boxes around detected persons

# Colors (BGR format for OpenCV)
BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (0, 255, 0)  # Green
BOX_THICKNESS = 2
FONT_SCALE = 0.6

# ESP32 Serial settings
ESP32_PORT = "/dev/ttyUSB0"  # Serial port for ESP32
ESP32_BAUDRATE = 115200  # Baud rate for serial communication
