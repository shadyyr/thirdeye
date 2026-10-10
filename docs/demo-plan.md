# Live demo plan

## Demo promise

Third Eye turns local visual perception into one useful tactile cue instead of flooding a wearer with every visible object. The memorable moment is that the wearer can point toward a person they are not watching on the monitor.

## Rehearsal sequence

1. Start the local dashboard and confirm both camera labels show healthy input.
2. Confirm the ESP32 serial controller is connected and the belt passes a short `STOP`/zone test.
3. Put one teammate in the belt without looking at the monitor.
4. Have another teammate enter the front camera from the right or stand behind the rig.
5. Show the dashboard path: detection box → track ID → motion state → relevance → selected haptic zone/intensity.
6. Let the wearer identify the direction from the belt feedback.
7. Move the person away and show relevance fall or a safe `STOP` command.
8. Repeat from the rear to demonstrate the second camera and calibration.

## Before judges arrive

- Verify front/rear orientation against the physical belt.
- Check that the selected event does not oscillate between cameras.
- Set the alert threshold so stationary background people do not dominate.
- Confirm serial watchdog disables all motors when the Jetson process stops.
- Hide nonessential terminal windows; keep the dashboard and a concise architecture visual visible.
- Prepare a short local video replay only as a debugging fallback, not as the primary demo.

## Recovery plan

- If one camera fails, demonstrate a reliable single-camera directional cue and say that the second stream is an integration extension.
- If serial transport fails, demonstrate the visible mock haptic command only after explaining that the hardware path is being isolated; do not claim belt feedback occurred.
- If live detection becomes unstable, switch back to the known working lightweight local model/configuration rather than adding features during judging.
