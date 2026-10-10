# ESP32 haptic controller

`firmware/third_eye_haptics.ino` is the initial Arduino sketch for the belt controller.

It receives UTF-8/ASCII lines over USB serial at 115200 baud:

```text
RIGHT,82
STOP,0
```

The sketch validates each command, drives only the requested zone with PWM, and stops all motors after the configurable 600 ms communication watchdog expires. Pin assignments live in one block at the top of the sketch so the physical wiring can change without touching protocol logic.

Important electrical constraints:

- ESP32 GPIO controls MOSFET gates only; never power motors from a GPIO pin.
- Use a suitable motor supply, shared ground between the ESP32 and the motor circuit, gate pulldown resistors, and flyback protection appropriate to the motor driver circuit.
- Start with one motor and low PWM intensity before installing all four belt zones.
- Set the pin constants to the actual wiring before flashing.

The Jetson-side software defaults to the mock transport. Once USB serial is verified, use a local configuration override with `"transport": "serial"`, the detected serial port, and the same baud rate.
