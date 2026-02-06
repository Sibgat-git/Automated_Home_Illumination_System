import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gpiozero import LED
import threading
import time

# --- CONFIGURATION ---
ZONE1_NAME = "Zone 1"
ZONE2_NAME = "Zone 2"
SCORE_THRESHOLD = 0.50
FRAME_SKIP = 2             
AI_SIZE = (320, 320)       
CONFIRMATION_FRAMES = 3    
PERSISTENCE_FRAMES = 15   

# 1. Initialize LEDs
green_led = LED(18)
red_led = LED(24)

# 2. Setup Detector
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
# FIX: Removed the accidental 'import cv2' from the middle of the function name
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=SCORE_THRESHOLD,
    category_allowlist=['person']
)
detector = vision.ObjectDetector.create_from_options(options)

# 3. Threaded Camera
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.success, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.success, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# 4. State Management
confirm_z1, timer_z1 = 0, 0
confirm_z2, timer_z2 = 0, 0
frame_count = 0

vs = VideoStream().start()
time.sleep(2.0) 

try:
    while True:
        frame = vs.read()
        if frame is None: break
        
        frame_count += 1
        h, w, _ = frame.shape
        mid_x = w // 2

        if frame_count % FRAME_SKIP == 0:
            small_frame = cv2.resize(frame, AI_SIZE)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
            
            results = detector.detect(mp_image)
            
            valid_z1 = False
            valid_z2 = False

            for det in results.detections:
                bbox = det.bounding_box
                scale_x, scale_y = w / AI_SIZE[0], h / AI_SIZE[1]
                
                real_x = bbox.origin_x * scale_x
                real_w = bbox.width * scale_x
                real_y = bbox.origin_y * scale_y
                real_h = bbox.height * scale_y
                
                # --- DETECTION LOGIC ---
                # Case A: Person is strictly in Zone 1
                if (real_x + real_w) <= mid_x:
                    valid_z1 = True
                    cv2.rectangle(frame, (int(real_x), int(real_y)), 
                                  (int(real_x + real_w), int(real_y + real_h)), (0, 255, 0), 2)
                
                # Case B: Person is strictly in Zone 2
                elif real_x >= mid_x:
                    valid_z2 = True
                    cv2.rectangle(frame, (int(real_x), int(real_y)), 
                                  (int(real_x + real_w), int(real_y + real_h)), (0, 0, 255), 2)
                
                # Case C: Person is overlapping BOTH
                elif real_x < mid_x and (real_x + real_w) > mid_x:
                    valid_z1 = True
                    valid_z2 = True
                    cv2.rectangle(frame, (int(real_x), int(real_y)), 
                                  (int(real_x + real_w), int(real_y + real_h)), (0, 255, 255), 2)

            # Update Persistence Timers
            if valid_z1:
                confirm_z1 += 1
                if confirm_z1 >= CONFIRMATION_FRAMES: timer_z1 = PERSISTENCE_FRAMES
            else:
                confirm_z1 = 0
                if timer_z1 > 0: timer_z1 -= 1

            if valid_z2:
                confirm_z2 += 1
                if confirm_z2 >= CONFIRMATION_FRAMES: timer_z2 = PERSISTENCE_FRAMES
            else:
                confirm_z2 = 0
                if timer_z2 > 0: timer_z2 -= 1

        # LED CONTROL
        if timer_z1 > 0: green_led.on()
        else: green_led.off()
            
        if timer_z2 > 0: red_led.on()
        else: red_led.off()

        # --- UI OVERLAY ---
        cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 255), 1)
        
        z1_label = f"{ZONE1_NAME}: {'DETECTED' if timer_z1 > 0 else 'CLEAR'}"
        z2_label = f"{ZONE2_NAME}: {'DETECTED' if timer_z2 > 0 else 'CLEAR'}"
        cv2.putText(frame, z1_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, z2_label, (mid_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if timer_z1 > 0 and timer_z2 > 0:
            msg = "PERSON DETECTED ON BOTH"
            t_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, msg, ((w - t_size[0]) // 2, h - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

        cv2.imshow('Fast Named-Zone Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    vs.stop()
    green_led.off(); red_led.off()
    cv2.destroyAllWindows()