import cv2
import time
import sys
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gpiozero import LED

# --- GET LIGHT DATA FROM MANAGER ---
# If script is run manually, default to 0 (dark)
try:
    lux_z1 = float(sys.argv[1]) if len(sys.argv) > 1 else 0
    lux_z2 = float(sys.argv[2]) if len(sys.argv) > 2 else 0
except:
    lux_z1, lux_z2 = 0, 0

# --- CONFIGURATION ---
LIGHT_THRESHOLD = 10
Z1_TIMEOUT = 30 
Z2_TIMEOUT = 30 
GLOBAL_EXIT_TIMEOUT = 45 

green_led = LED(18) # LED for Zone 1
red_led = LED(24)   # LED for Zone 2

# AI Setup
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    category_allowlist=['person']
)
detector = vision.ObjectDetector.create_from_options(options)

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

vs = VideoStream().start()
time.sleep(2.0)

last_seen_z1 = 0
last_seen_z2 = 0
last_any_seen = time.time()

try:
    while True:
        frame = vs.read()
        if frame is None: break
        h, w, _ = frame.shape
        mid_x = w // 2
        now = time.time()

        # Run AI Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        for det in results.detections:
            bbox = det.bounding_box
            rx, rw = int(bbox.origin_x), int(bbox.width)
            
            # Logic: Update timers ONLY if the specific zone is dark
            # This allows BOTH to be active if both sensors report low light
            if rx < mid_x and lux_z1 < LIGHT_THRESHOLD:
                last_seen_z1 = now
            if (rx + rw) > mid_x and lux_z2 < LIGHT_THRESHOLD:
                last_seen_z2 = now
            
            last_any_seen = now
            cv2.rectangle(frame, (rx, int(bbox.origin_y)), (rx+rw, int(bbox.origin_y+bbox.height)), (255, 255, 0), 2)

        # --- COUNTDOWNS ---
        z1_rem = max(0, int(Z1_TIMEOUT - (now - last_seen_z1))) if last_seen_z1 > 0 else 0
        z2_rem = max(0, int(Z2_TIMEOUT - (now - last_seen_z2))) if last_seen_z2 > 0 else 0
        shutdown_rem = max(0, int(GLOBAL_EXIT_TIMEOUT - (now - last_any_seen)))

        # --- LED CONTROL ---
        # Zone 1 (Green)
        if z1_rem > 0:
            green_led.on()
            z1_txt = f"Z1: {z1_rem}s"
        else:
            green_led.off()
            z1_txt = "Z1: DARK/EMPTY" if lux_z1 < LIGHT_THRESHOLD else "Z1: BRIGHT"

        # Zone 2 (Red)
        if z2_rem > 0:
            red_led.on()
            z2_txt = f"Z2: {z2_rem}s"
        else:
            red_led.off()
            z2_txt = "Z2: DARK/EMPTY" if lux_z2 < LIGHT_THRESHOLD else "Z2: BRIGHT"

        # --- UI & TERMINAL ---
        print(f"\r[Timers] {z1_txt} | {z2_txt} | EXIT: {shutdown_rem}s      ", end="")
        cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)
        cv2.putText(frame, z1_txt, (20, 50), 2, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, z2_txt, (mid_x + 20, 50), 2, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"SHUTDOWN IN: {shutdown_rem}s", (w//4, h-30), 2, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Dual-Zone Lighting Monitor", frame)
        
        # Exit if 'q' is pressed or if no one is seen for GLOBAL_EXIT_TIMEOUT
        if cv2.waitKey(1) & 0xFF == ord('q') or shutdown_rem <= 0:
            break

finally:
    vs.stop()
    green_led.off()
    red_led.off()
    cv2.destroyAllWindows()
    sys.exit(0)