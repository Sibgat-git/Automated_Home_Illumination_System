"""
Camera handling module for Raspberry Pi.
Supports USB webcams and Pi Camera Module.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class Camera:
    """Handles camera capture for person detection."""

    def __init__(
        self,
        camera_index: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30
    ):
        """
        Initialize the camera.

        Args:
            camera_index: Camera device index (0 for default camera)
            resolution: Capture resolution as (width, height)
            fps: Target frames per second
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        """
        Start the camera capture.

        Returns:
            True if camera started successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"Camera started: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def stop(self) -> None:
        """Release the camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
