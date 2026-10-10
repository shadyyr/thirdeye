import csv
import math
import time
from collections import deque

import cv2
from ultralytics import YOLO


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

LEFT_BOUNDARY = 0.33
RIGHT_BOUNDARY = 0.67

CONFIDENCE_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Motion estimation tuning (all values assume roughly 30 FPS)
# ---------------------------------------------------------------------------

# EMA smoothing factor for the bounding-box area (0..1).
# Higher = reacts faster but noisier; lower = smoother but more lag.
EMA_ALPHA = 0.30

# Number of smoothed-area samples used to measure the trend (~0.33 s at 30 FPS).
MOTION_WINDOW_FRAMES = 10

# Minimum samples in the window before any motion decision is made.
MOTION_MIN_SAMPLES = 8

# Hysteresis thresholds on the trend (fractional area change across the window).
# Enter APPROACHING / MOVING AWAY when |trend| rises above ENTER;
# drop back to STABLE only when |trend| falls below EXIT.
MOTION_ENTER_THRESHOLD = 0.06
MOTION_EXIT_THRESHOLD = 0.03

# Consecutive frames a new state must be indicated before switching to it.
MOTION_ENTER_CONFIRM_FRAMES = 2
MOTION_EXIT_CONFIRM_FRAMES = 3

# Frames without a person tolerated before the motion history is reset.
MOTION_MAX_MISSING_FRAMES = 5

# If the raw area jumps by more than this ratio vs. the smoothed area in one
# frame, assume the "largest person" switched to someone else and reset.
MOTION_JUMP_RESET_RATIO = 1.8

# Diagnostic CSV logging
MOTION_LOG_ENABLED = True
MOTION_LOG_PATH = "motion_debug.csv"
MOTION_LOG_EVERY_N_FRAMES = 1   # 1 = every frame with a person
MOTION_LOG_FLUSH_EVERY = 60     # rows buffered before flushing to disk

# ---------------------------------------------------------------------------
# Haptic command generation (software only, no ESP32 output yet)
# ---------------------------------------------------------------------------

# Minimum YOLO detection confidence of the selected person required to emit a
# haptic command. Separate from CONFIDENCE_THRESHOLD: detections between the
# two are still drawn, they just do not produce a haptic command.
HAPTIC_CONFIDENCE_THRESHOLD = 0.50

HAPTIC_NONE = "NONE"

# Motion state -> urgency level. States not listed here produce NONE.
HAPTIC_URGENCY_BY_MOTION = {
    "APPROACHING": "HIGH",
    "STABLE": "LOW",
}

HAPTIC_DIRECTIONS = ("LEFT", "FRONT", "RIGHT")


def get_direction(center_x: float, frame_width: int) -> str:
    normalized_x = center_x / frame_width

    if normalized_x < LEFT_BOUNDARY:
        return "LEFT"

    if normalized_x > RIGHT_BOUNDARY:
        return "RIGHT"

    return "FRONT"


def get_haptic_command(direction, motion: str, confidence: float) -> str:
    """
    Convert the selected person's direction, motion state and detection
    confidence into a haptic command such as "LEFT_HIGH" or "NONE".

    Pure function with no camera / YOLO / GUI dependencies, so a future
    ESP32 communication layer can call it (or consume its output) directly.
    Pass direction=None when no person is selected.
    """
    if direction not in HAPTIC_DIRECTIONS:
        return HAPTIC_NONE

    if confidence < HAPTIC_CONFIDENCE_THRESHOLD:
        return HAPTIC_NONE

    urgency = HAPTIC_URGENCY_BY_MOTION.get(motion)

    if urgency is None:
        return HAPTIC_NONE

    return f"{direction}_{urgency}"


def haptic_color(command: str) -> tuple:
    # BGR colors for the on-screen haptic banner.
    if command.endswith("_HIGH"):
        return (0, 0, 255)

    if command.endswith("_LOW"):
        return (0, 200, 255)

    return (160, 160, 160)


class MotionEstimator:
    """
    Temporal approach/recede estimator for the selected person.

    1. Smooth the raw bounding-box area with an EMA.
    2. Keep the last MOTION_WINDOW_FRAMES smoothed areas.
    3. Fit a least-squares line to log(smoothed area) over the window and
       convert its slope to a fractional area change across the window.
    4. Apply hysteresis (enter/exit thresholds + confirm frames) to get a
       stable APPROACHING / STABLE / MOVING AWAY state.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.smoothed_area = None
        self.sample_count = 0
        self.history = deque(maxlen=MOTION_WINDOW_FRAMES)
        self.state = "UNKNOWN"
        self.candidate_state = None
        self.candidate_count = 0
        self.missing_frames = 0
        self.trend = 0.0
        self.confidence = 0.0

    def mark_missing(self):
        self.missing_frames += 1

        if self.missing_frames > MOTION_MAX_MISSING_FRAMES:
            self.reset()

    def update(self, raw_area: float) -> str:
        self.missing_frames = 0

        if raw_area <= 0:
            return self.state

        if self.smoothed_area is not None:
            jump = max(raw_area, self.smoothed_area) / min(raw_area, self.smoothed_area)

            if jump > MOTION_JUMP_RESET_RATIO:
                self.reset()

        # Bias-corrected EMA: behaves like a running mean for the first few
        # samples so the initial noisy measurement does not fake a trend.
        self.sample_count += 1
        alpha = max(EMA_ALPHA, 1.0 / self.sample_count)

        if self.smoothed_area is None:
            self.smoothed_area = raw_area
        else:
            self.smoothed_area = (
                alpha * raw_area + (1 - alpha) * self.smoothed_area
            )

        self.history.append(math.log(self.smoothed_area))

        if len(self.history) < MOTION_MIN_SAMPLES:
            self.trend = 0.0
            self.confidence = 0.0
            self.state = "UNKNOWN"
            return self.state

        self.trend = self._window_trend()
        self._apply_hysteresis(self._indicated_state())
        self.confidence = self._motion_confidence()

        return self.state

    def _window_trend(self) -> float:
        # Least-squares slope of log(area) per frame, scaled to the window
        # length and converted back to a fractional area change.
        n = len(self.history)
        mean_x = (n - 1) / 2
        mean_y = sum(self.history) / n

        numerator = 0.0
        denominator = 0.0

        for x, y in enumerate(self.history):
            numerator += (x - mean_x) * (y - mean_y)
            denominator += (x - mean_x) ** 2

        slope = numerator / denominator

        return math.exp(slope * (MOTION_WINDOW_FRAMES - 1)) - 1

    def _indicated_state(self) -> str:
        trend = self.trend

        if trend >= MOTION_ENTER_THRESHOLD:
            return "APPROACHING"

        if trend <= -MOTION_ENTER_THRESHOLD:
            return "MOVING AWAY"

        # Inside the hysteresis band: keep a moving state until the trend
        # drops below the exit threshold.
        if self.state == "APPROACHING" and trend > MOTION_EXIT_THRESHOLD:
            return "APPROACHING"

        if self.state == "MOVING AWAY" and trend < -MOTION_EXIT_THRESHOLD:
            return "MOVING AWAY"

        return "STABLE"

    def _apply_hysteresis(self, indicated: str):
        if self.state == "UNKNOWN":
            # Warm-up finished: start from STABLE so leaving it still has to
            # pass the normal confirm frames (avoids a noisy first decision).
            self.state = "STABLE"

        if indicated == self.state:
            self.candidate_state = None
            self.candidate_count = 0
            return

        if indicated == self.candidate_state:
            self.candidate_count += 1
        else:
            self.candidate_state = indicated
            self.candidate_count = 1

        if indicated == "STABLE":
            required = MOTION_EXIT_CONFIRM_FRAMES
        else:
            required = MOTION_ENTER_CONFIRM_FRAMES

        if self.candidate_count >= required:
            self.state = indicated
            self.candidate_state = None
            self.candidate_count = 0

    def _motion_confidence(self) -> float:
        fill = len(self.history) / MOTION_WINDOW_FRAMES
        magnitude = abs(self.trend)

        if self.state in ("APPROACHING", "MOVING AWAY"):
            score = min(1.0, magnitude / (2 * MOTION_ENTER_THRESHOLD))
        else:
            score = max(0.0, 1.0 - magnitude / MOTION_ENTER_THRESHOLD)

        return score * fill


class MotionLogger:
    """Buffered CSV writer for motion calibration data."""

    FIELDS = [
        "timestamp",
        "frame",
        "fps",
        "raw_area",
        "smoothed_area",
        "area_frac_of_frame",
        "trend_pct",
        "direction",
        "state",
        "motion_confidence",
        "detection_confidence",
    ]

    def __init__(self, path: str):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.FIELDS)
        self.rows_since_flush = 0

    def log(self, row: list):
        self.writer.writerow(row)
        self.rows_since_flush += 1

        if self.rows_since_flush >= MOTION_LOG_FLUSH_EVERY:
            self.file.flush()
            self.rows_since_flush = 0

    def close(self):
        self.file.flush()
        self.file.close()


pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    f"video/x-raw(memory:NVMM), width={FRAME_WIDTH}, "
    f"height={FRAME_HEIGHT}, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=true max-buffers=1"
)


cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("Could not open CSI camera")


model = YOLO("yolo26n.pt")

motion_estimator = MotionEstimator()
motion_logger = MotionLogger(MOTION_LOG_PATH) if MOTION_LOG_ENABLED else None

frame_index = 0
fps = 0.0
last_haptic_command = None
last_frame_time = time.monotonic()

print("Third Eye live detection started.")
print("Press Q in the camera window to quit.")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame")
        break

    frame_index += 1
    now = time.monotonic()
    frame_dt = now - last_frame_time
    last_frame_time = now

    if frame_dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / frame_dt) if fps else 1.0 / frame_dt

    results = model.predict(
        frame,
        device=0,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]
    annotated = result.plot()

    detected_people = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        if class_name != "person":
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        width = x2 - x1
        height = y2 - y1
        area = width * height

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        detected_people.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y,
                "area": area,
                "confidence": confidence,
            }
        )

    haptic_command = HAPTIC_NONE
    haptic_note = "no person"

    # For now, choose the largest visible person.
    # Usually this corresponds to the nearest / most relevant person.
    if detected_people:
        target = max(detected_people, key=lambda person: person["area"])

        center_x = target["center_x"]
        center_y = target["center_y"]
        area = target["area"]
        confidence = target["confidence"]

        direction = get_direction(center_x, frame.shape[1])
        motion = motion_estimator.update(area)

        haptic_command = get_haptic_command(direction, motion, confidence)

        if confidence < HAPTIC_CONFIDENCE_THRESHOLD:
            haptic_note = f"conf {confidence:.2f} < {HAPTIC_CONFIDENCE_THRESHOLD:.2f}"
        else:
            haptic_note = f"{direction} | {motion} | conf {confidence:.2f}"

        print(
            f"PERSON | {direction:<5} | "
            f"{motion:<11} | "
            f"trend={motion_estimator.trend * 100:+6.1f}% | "
            f"confidence={confidence:.2f}"
        )

        if motion_logger and frame_index % MOTION_LOG_EVERY_N_FRAMES == 0:
            smoothed = motion_estimator.smoothed_area or 0.0
            frame_area = frame.shape[0] * frame.shape[1]

            motion_logger.log(
                [
                    f"{time.time():.3f}",
                    frame_index,
                    f"{fps:.1f}",
                    f"{area:.0f}",
                    f"{smoothed:.0f}",
                    f"{area / frame_area:.4f}",
                    f"{motion_estimator.trend * 100:.2f}",
                    direction,
                    motion,
                    f"{motion_estimator.confidence:.2f}",
                    f"{confidence:.2f}",
                ]
            )

        cv2.circle(
            annotated,
            (int(center_x), int(center_y)),
            7,
            (255, 255, 255),
            -1
        )

        label = f"{direction} | {motion}"

        cv2.putText(
            annotated,
            label,
            (
                int(target["x1"]),
                max(30, int(target["y1"]) - 35)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    else:
        motion_estimator.mark_missing()

    # Terminal: only report when the haptic command changes.
    if haptic_command != last_haptic_command:
        print(f">>> HAPTIC {haptic_command:<10} ({haptic_note})")
        last_haptic_command = haptic_command

    # GUI: prominent haptic command banner at the bottom of the frame.
    banner_color = haptic_color(haptic_command)
    banner_top = frame.shape[0] - 80

    cv2.rectangle(
        annotated,
        (0, banner_top),
        (frame.shape[1], frame.shape[0]),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated,
        f"HAPTIC: {haptic_command}",
        (20, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        banner_color,
        4
    )

    cv2.putText(
        annotated,
        haptic_note,
        (frame.shape[1] - 520, frame.shape[0] - 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Draw LEFT / FRONT / RIGHT zone boundaries
    left_x = int(frame.shape[1] * LEFT_BOUNDARY)
    right_x = int(frame.shape[1] * RIGHT_BOUNDARY)

    cv2.line(
        annotated,
        (left_x, 0),
        (left_x, frame.shape[0]),
        (255, 255, 255),
        2
    )

    cv2.line(
        annotated,
        (right_x, 0),
        (right_x, frame.shape[0]),
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        "LEFT",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        "FRONT",
        (frame.shape[1] // 2 - 60, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        "RIGHT",
        (frame.shape[1] - 150, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Third Eye - Spatial Awareness", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


if motion_logger:
    motion_logger.close()

cap.release()
cv2.destroyAllWindows()
