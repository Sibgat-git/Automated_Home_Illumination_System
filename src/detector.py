"""
Person detector using TensorFlow Lite.
Optimized for Raspberry Pi 4.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


class Detection:
    """Represents a single detection result."""

    def __init__(
        self,
        class_id: int,
        label: str,
        confidence: float,
        bbox: Tuple[int, int, int, int]
    ):
        """
        Args:
            class_id: Class ID from the model
            label: Human-readable label
            confidence: Detection confidence (0-1)
            bbox: Bounding box as (x_min, y_min, x_max, y_max)
        """
        self.class_id = class_id
        self.label = label
        self.confidence = confidence
        self.bbox = bbox

    @property
    def x_min(self) -> int:
        return self.bbox[0]

    @property
    def y_min(self) -> int:
        return self.bbox[1]

    @property
    def x_max(self) -> int:
        return self.bbox[2]

    @property
    def y_max(self) -> int:
        return self.bbox[3]

    def __repr__(self) -> str:
        return f"Detection({self.label}, {self.confidence:.2f}, {self.bbox})"


class PersonDetector:
    """TensorFlow Lite person detector for Raspberry Pi."""

    # COCO dataset class ID for person
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        num_threads: int = 4
    ):
        """
        Initialize the person detector.

        Args:
            model_path: Path to the TFLite model file
            labels_path: Path to labels file (optional)
            confidence_threshold: Minimum confidence for detections
            num_threads: Number of CPU threads for inference
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.num_threads = num_threads

        # Load labels
        self.labels = self._load_labels(labels_path)

        # Initialize interpreter
        self.interpreter = Interpreter(
            model_path=str(self.model_path),
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get model input dimensions
        input_shape = self.input_details[0]['shape']
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        print(f"Model loaded: {self.model_path.name}")
        print(f"Input size: {self.input_width}x{self.input_height}")

    def _load_labels(self, labels_path: Optional[str]) -> dict:
        """Load labels from file or use default COCO labels."""
        if labels_path and Path(labels_path).exists():
            labels = {}
            with open(labels_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        labels[int(parts[0])] = parts[1]
                    elif len(parts) == 1:
                        # Label file without IDs (one label per line)
                        labels[len(labels)] = parts[0]
            return labels

        # Default: just person label (class 0 in COCO SSD models)
        return {0: "person"}

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Preprocessed tensor ready for inference
        """
        import cv2

        # Resize to model input size
        resized = cv2.resize(frame, (self.input_width, self.input_height))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Check if model expects float or uint8
        input_dtype = self.input_details[0]['dtype']

        if input_dtype == np.float32:
            # Normalize to [0, 1] or [-1, 1] based on model
            processed = rgb.astype(np.float32) / 255.0
        else:
            # Keep as uint8
            processed = rgb.astype(np.uint8)

        # Add batch dimension
        return np.expand_dims(processed, axis=0)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run person detection on a frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of Detection objects for detected persons
        """
        # Get original frame dimensions for bbox scaling
        orig_height, orig_width = frame.shape[:2]

        # Preprocess
        input_tensor = self.preprocess(frame)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        # Get outputs (format depends on model)
        # Standard SSD MobileNet output format:
        # - boxes: [1, num_detections, 4] - normalized coordinates
        # - classes: [1, num_detections]
        # - scores: [1, num_detections]
        # - count: number of detections

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        detections = []

        for i in range(len(scores)):
            if scores[i] < self.confidence_threshold:
                continue

            class_id = int(classes[i])

            # Only detect persons (class 0 in COCO)
            if class_id != self.PERSON_CLASS_ID:
                continue

            # Convert normalized coordinates to pixel coordinates
            # Box format: [y_min, x_min, y_max, x_max] (normalized)
            y_min, x_min, y_max, x_max = boxes[i]

            bbox = (
                int(x_min * orig_width),
                int(y_min * orig_height),
                int(x_max * orig_width),
                int(y_max * orig_height)
            )

            label = self.labels.get(class_id, f"class_{class_id}")

            detections.append(Detection(
                class_id=class_id,
                label=label,
                confidence=float(scores[i]),
                bbox=bbox
            ))

        return detections

    def detect_persons(self, frame: np.ndarray) -> Tuple[bool, List[Detection]]:
        """
        Detect if any persons are present in the frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Tuple of (person_detected, list of detections)
        """
        detections = self.detect(frame)
        return len(detections) > 0, detections
