# Judging strategy

## Core story

Third Eye is not an object detector attached to a belt. It is a local physical-AI loop that senses a real environment, decides which spatial event matters, and communicates that result non-visually through a wearable interface.

## Demonstrate, then explain

Lead with the belt moment. After the wearer points in the correct direction, use the dashboard to explain the visible chain: two cameras → local Jetson model → tracking → motion and salience → USB serial → ESP32/MOSFET belt output.

## Challenge alignment

| Target | Evidence to show |
| --- | --- |
| NVIDIA / Jetson | Jetson is running the local perception pipeline on camera streams with no cloud vision dependency. |
| Embedded Hack | ESP32 firmware, watchdog behavior, MOSFET-switched motors, and a wearable four-zone output. |
| AI Hack | Detection is only the first step; tracking, motion, and transparent event scoring drive the output. |
| Encompass | Cameras collect physical-world spatial data and the system transforms it into tactile awareness. |
| Open source AI | Only claim this if the final locally run model is documented with its source, license, and real role. |

## What to avoid claiming

- Exact distance or 3D localization from two camera feeds.
- Full scene understanding, facial recognition, or all-object awareness.
- Cloud inference or TensorRT acceleration unless those are genuinely running.
- A working eight-zone belt, mobile app, or other stretch goal that has not been rehearsed.

## Technical talking points

- Separate front/rear streams avoid brittle panoramic stitching.
- Bounding-box-area trends provide a pragmatic, transparent approximation of approach/retreat.
- A relevance threshold and selection hold prevent notification overload and rapid switching.
- USB serial and ESP32 watchdog behavior make the physical output reliable and fail-safe for a hackathon floor.
