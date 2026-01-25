#!/usr/bin/env python3
"""
Automated Home Illumination System - Person Detection
Main entry point for running person detection on Raspberry Pi 4.
"""

import cv2
import time
import argparse
import subprocess
from pathlib import Path

import config
from src.camera import Camera
from src.detector import PersonDetector


def get_available_cameras():
    """Detect available cameras on the system."""
    cameras = []

    # Try using v4l2-ctl for detailed info (Linux)
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            current_name = None
            for line in lines:
                if not line.startswith('\t') and line.strip():
                    current_name = line.strip().rstrip(':')
                elif line.strip().startswith('/dev/video'):
                    device = line.strip()
                    index = int(device.replace('/dev/video', ''))
                    # Test if camera is a capture device
                    cap = cv2.VideoCapture(index)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            cameras.append({
                                'index': index,
                                'name': current_name or f"Camera {index}",
                                'device': device
                            })
                        cap.release()
            if cameras:
                return cameras
    except FileNotFoundError:
        pass

    # Fallback: probe camera indices directly
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append({
                    'index': i,
                    'name': f"Camera {i}",
                    'device': f"/dev/video{i}"
                })
            cap.release()

    return cameras


def list_cameras():
    """Print available cameras."""
    print("\nDetecting available cameras...\n")
    cameras = get_available_cameras()

    if not cameras:
        print("No cameras found.")
        return []

    print(f"Found {len(cameras)} camera(s):\n")
    print("-" * 50)
    for cam in cameras:
        print(f"  Index {cam['index']}: {cam['name']}")
        print(f"           Device: {cam['device']}")
        print()
    print("-" * 50)

    return cameras


def select_camera():
    """Interactive camera selection."""
    cameras = list_cameras()

    if not cameras:
        return None

    if len(cameras) == 1:
        print(f"Using the only available camera (index {cameras[0]['index']})")
        return cameras[0]['index']

    while True:
        try:
            choice = input("\nEnter camera index to use (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None

            index = int(choice)
            valid_indices = [cam['index'] for cam in cameras]

            if index in valid_indices:
                return index
            else:
                print(f"Invalid index. Choose from: {valid_indices}")
        except ValueError:
            print("Please enter a valid number.")


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    for det in detections:
        # Draw bounding box
        cv2.rectangle(
            frame,
            (det.x_min, det.y_min),
            (det.x_max, det.y_max),
            config.BOX_COLOR,
            config.BOX_THICKNESS
        )

        # Draw label with confidence
        label = f"{det.label}: {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            1
        )

        # Background for text
        cv2.rectangle(
            frame,
            (det.x_min, det.y_min - label_size[1] - 10),
            (det.x_min + label_size[0], det.y_min),
            config.BOX_COLOR,
            -1
        )

        # Text
        cv2.putText(
            frame,
            label,
            (det.x_min, det.y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            (0, 0, 0),
            1
        )

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Person Detection for Automated Home Illumination"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.MODEL_PATH),
        help="Path to TFLite model"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index (if not specified, will prompt for selection)"
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window"
    )
    args = parser.parse_args()

    # List cameras and exit if requested
    if args.list_cameras:
        list_cameras()
        return 0

    # Select camera if not specified
    if args.camera is None:
        camera_index = select_camera()
        if camera_index is None:
            print("No camera selected. Exiting.")
            return 1
    else:
        camera_index = args.camera

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Run 'python download_model.py' to download the model.")
        return 1

    # Initialize detector
    print("Initializing person detector...")
    detector = PersonDetector(
        model_path=args.model,
        confidence_threshold=args.threshold,
        num_threads=config.NUM_THREADS
    )

    # Initialize camera
    print(f"Starting camera (index {camera_index})...")
    camera = Camera(
        camera_index=camera_index,
        resolution=config.CAMERA_RESOLUTION,
        fps=config.CAMERA_FPS
    )

    if not camera.start():
        return 1

    show_preview = config.SHOW_PREVIEW and not args.no_preview

    print("\nPerson detection running. Press 'q' to quit.\n")

    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    try:
        while True:
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Run detection
            person_detected, detections = detector.detect_persons(frame)

            # Update FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # Log detection
            if person_detected:
                print(f"[{time.strftime('%H:%M:%S')}] Person detected! "
                      f"(count: {len(detections)}, FPS: {fps:.1f})")

            # Draw and show preview
            if show_preview:
                if config.DRAW_BOUNDING_BOXES:
                    frame = draw_detections(frame, detections)

                # Draw FPS
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    config.TEXT_COLOR,
                    2
                )

                # Draw detection status
                status = "PERSON DETECTED" if person_detected else "No person"
                color = (0, 0, 255) if person_detected else (0, 255, 0)
                cv2.putText(
                    frame,
                    status,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

                cv2.imshow("Person Detection", frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        camera.stop()
        if show_preview:
            cv2.destroyAllWindows()

    print("Done.")
    return 0


if __name__ == "__main__":
    exit(main())
