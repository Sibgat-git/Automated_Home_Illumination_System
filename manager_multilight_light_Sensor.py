import smbus2
import time
import subprocess
from gpiozero import MotionSensor

# --- CONFIGURATION ---
PIR_PIN = 27         
LIGHT_THRESHOLD = 10  
COOLDOWN_SECONDS = 5  
ADDR_Z1 = 0x23  # ADDR Pin to GND
ADDR_Z2 = 0x5C  # ADDR Pin to 3.3V

bus = smbus2.SMBus(1)

def read_light(addr):
    """Reads light level from a specific I2C address."""
    try:
        bus.write_byte(addr, 0x01) # Power On
        bus.write_byte(addr, 0x20) # High Res Mode
        time.sleep(0.2)
        data = bus.read_i2c_block_data(addr, 0x00, 2)
        return round((data[0] << 8 | data[1]) / 1.2, 2)
    except Exception:
        return None

pir = MotionSensor(PIR_PIN)

print("--- Dual-Zone Light Manager Active ---")

try:
    while True:
        # Wait for physical motion first
        pir.wait_for_motion()
        
        lux_1 = read_light(ADDR_Z1)
        lux_2 = read_light(ADDR_Z2)
        
        # Determine if each zone is dark based on THRESHOLD
        z1_dark = lux_1 is not None and lux_1 < LIGHT_THRESHOLD
        z2_dark = lux_2 is not None and lux_2 < LIGHT_THRESHOLD
        
        # Trigger AI if Zone 1 IS DARK, OR Zone 2 IS DARK, or BOTH are DARK
        if z1_dark or z2_dark:
            print(f"🌙 Motion + Low Light (Z1:{lux_1}lx, Z2:{lux_2}lx). Launching AI...")
            # Pass the light values as strings so ai_detector.py can process them
            subprocess.run(["python3", "ai_detector.py", str(lux_1), str(lux_2)])
            
            # Cooldown to prevent immediate re-triggering after AI script exits
            time.sleep(COOLDOWN_SECONDS)
        else:
            print(f"☀️ Motion detected, but room is bright (Z1:{lux_1}lx, Z2:{lux_2}lx).")
            time.sleep(2)

except KeyboardInterrupt:
    print("\nShutting down Manager...")