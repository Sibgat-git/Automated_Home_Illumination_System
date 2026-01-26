"""
ESP32 MicroPython code for LED control based on person detection.
Upload this file to your ESP32 as 'main.py' using Thonny or ampy.

Wiring:
  - Green LED: GPIO 2 -> LED anode (+) -> 220 ohm resistor -> GND
  - Yellow LED: GPIO 4 -> LED anode (+) -> 220 ohm resistor -> GND
"""

from machine import Pin
import sys
import uselect
import time

# LED GPIO pins (change these if using different pins)
GREEN_LED_PIN = 2   # Person detected
YELLOW_LED_PIN = 4  # No person detected

# Initialize LEDs
green_led = Pin(GREEN_LED_PIN, Pin.OUT)
yellow_led = Pin(YELLOW_LED_PIN, Pin.OUT)

# Start with yellow LED on (no person detected initially)
green_led.off()
yellow_led.on()

print("ESP32 LED Controller Ready")
print("Green LED: GPIO", GREEN_LED_PIN)
print("Yellow LED: GPIO", YELLOW_LED_PIN)
print("Waiting for commands...")

# Set up polling for stdin
poll = uselect.poll()
poll.register(sys.stdin, uselect.POLLIN)

buffer = ""

while True:
    # Check for incoming data with 100ms timeout
    events = poll.poll(100)

    if events:
        char = sys.stdin.read(1)
        if char:
            if char == '\n' or char == '\r':
                command = buffer.strip()
                buffer = ""

                if command == "PERSON":
                    green_led.on()
                    yellow_led.off()
                    print("LED: GREEN")

                elif command == "NONE":
                    green_led.off()
                    yellow_led.on()
                    print("LED: YELLOW")

                elif command == "PING":
                    print("PONG")

                elif command == "STATUS":
                    g = "ON" if green_led.value() else "OFF"
                    y = "ON" if yellow_led.value() else "OFF"
                    print("Green:", g, "Yellow:", y)
            else:
                buffer += char
