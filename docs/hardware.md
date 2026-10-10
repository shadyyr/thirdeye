# Hardware plan

## MVP topology

```text
Jetson Orin Nano + front/rear camera rig
        │ USB serial command: ZONE,INTENSITY
        ▼
ESP32 GPIO pins ──> MOSFET gate inputs ──> 3V coin vibration motors
                                                ▲
                                        suitable motor power path
```

The Jetson and cameras remain near the table display. The wearable side is the belt, four motors, ESP32, and suitable motor power arrangement. This keeps CSI ribbon cable length and user comfort out of the MVP critical path.

## Four belt zones

Place one motor at each consistent body-relative location:

- `FRONT`
- `RIGHT`
- `REAR`
- `LEFT`

Label the physical motor leads and verify one zone at a time before attaching the belt. Confirm camera orientation against the wearer before locking in the rear left/right mapping.

## Electrical checklist

- An ESP32 GPIO drives only a MOSFET gate; it must not power a motor directly.
- Use a common ground between ESP32 logic and the motor driver supply.
- Use gate pulldown resistors so motors stay off during reset/boot.
- Add flyback protection appropriate for the motor driver arrangement.
- Check motor voltage/current requirements and wire gauge before selecting the supply.
- Test one motor at low PWM duty cycle, then check heat, voltage sag, and radio/serial stability before scaling to four.
- Secure joints and add strain relief before the live demo.

The exact pin assignments currently live only in the ESP32 sketch. Update the four constants there after wiring; do not duplicate them across firmware files.

## Communication behavior

The planned MVP transport is direct USB serial at 115200 baud. This avoids a Wi-Fi dependency. The ESP32 accepts strict lines such as `RIGHT,82` and `STOP,0`, emits serial debug responses, and stops all motors if it does not receive a valid command within 600 ms.

## Integration order

1. Flash the firmware and confirm serial debug output with no motors attached.
2. Attach one MOSFET-controlled motor and verify `FRONT,25` then `STOP,0`.
3. Verify watchdog stop by disconnecting or pausing the command stream.
4. Add and label the remaining three zones.
5. Connect the Jetson serial adapter only after the standalone controller is safe and repeatable.
