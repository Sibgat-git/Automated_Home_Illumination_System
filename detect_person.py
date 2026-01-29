import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gpiozero import LED

# 1. Initialize LEDs on GPIO pins
# Green LED (GPIO 18 / Physical Pin 12)
# Red LED (GPIO 24 / Physical Pin 18)
green_led = LED(18)
red_led = LED(24)

# 2. Setup MediaPipe Object Detector
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    category_allowlist=['person']
)
detector = vision.ObjectDetector.create_from_options(options)

# 3. Start Webcam
cap = cv2.VideoCapture(0)

print("Starting Detection... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Run detection
    detection_result = detector.detect(mp_image)
    
    person_count = len(detection_result.detections)

    if person_count > 0:
        # --- PERSON DETECTED LOGIC ---
        green_led.on()
        red_led.off()
        
        # Display label at top left
        cv2.putText(frame, "PERSON DETECTED", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Draw individual boxes for every person found
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
            
            # Green bounding box
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
    else:
        # --- NO PERSON DETECTED LOGIC ---
        green_led.off()
        red_led.on()
        
        # Display label at top left
        cv2.putText(frame, "NO PERSON DETECTED", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the video feed
    cv2.imshow('Automated Home Illumination System', frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("Closing application...")
green_led.off()
red_led.off()
cap.release()
cv2.destroyAllWindows()