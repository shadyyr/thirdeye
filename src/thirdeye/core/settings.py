"""All Third Eye event-platform tunables in one place.

Edit values here during rehearsal. Times are seconds; "ratio" distances are
relative to the person's bounding-box height so they work at any range.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Camera (identical to the known-good live_yolo.py pipeline)
# ---------------------------------------------------------------------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    f"video/x-raw(memory:NVMM), width={FRAME_WIDTH}, "
    f"height={FRAME_HEIGHT}, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=true max-buffers=1"
)

# ---------------------------------------------------------------------------
# Detector (YOLO26n on CUDA, same model/imgsz as live_yolo.py)
# ---------------------------------------------------------------------------
MODEL_PATH = "yolo26n.pt"
YOLO_IMGSZ = 640
YOLO_DEVICE = 0
# Floor passed to YOLO. Low-confidence boxes only extend existing tracks.
YOLO_MIN_CONFIDENCE = 0.15
# COCO class ids: person=0, backpack=24, handbag=26, suitcase=28
YOLO_CLASS_IDS = [0, 24, 26, 28]

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
# Confidence needed to START a new track (per class). Lower-confidence
# detections may still update an existing track.
TRACK_NEW_CONFIDENCE = {"person": 0.40, "backpack": 0.25, "handbag": 0.25, "suitcase": 0.25}
# Consecutive matched frames before a new track is reported (filters flicker).
TRACK_MIN_HITS = 3
# Seconds a confirmed track may go undetected before it is dropped
# (time-based so behavior is the same at 15 or 22 FPS).
TRACK_MAX_MISSED_PERSON_SECONDS = 1.0
TRACK_MAX_MISSED_OBJECT_SECONDS = 4.0  # bags flicker more and are usually still
# Low-confidence detections may only extend a track that was detected within
# this many seconds (stops YOLO duplicate boxes from keeping ghosts alive).
TRACK_LOW_CONF_EXTEND_SECONDS = 0.3
# Matching gates
TRACK_MIN_IOU = 0.15
# Max center distance for a match, as a fraction of the track's box diagonal.
TRACK_MAX_CENTER_DIST_RATIO = 0.75
# Spatial-persistence fallback for objects: a lost object that reappears
# within this many seconds near its last position keeps its old ID.
TRACK_OBJECT_REVIVE_SECONDS = 15.0
TRACK_OBJECT_REVIVE_DIST_RATIO = 1.0  # fraction of the old box diagonal

# Confirmed same-family tracks that overlap this much for TRACK_MERGE_SECONDS
# are duplicates of one entity: the newer track is merged into the older one.
TRACK_MERGE_SECONDS = 0.5
TRACK_MERGE_CONTAIN = 0.85
TRACK_MERGE_MIN_IOU = 0.4

# ---------------------------------------------------------------------------
# Duplicate-box filter (before tracking). Derived from rehearsal.mp4: YOLO26n
# emits a second, nested box on one person (upper-body or person+bag box) and
# labels one bag as backpack + handbag + suitcase at once.
# ---------------------------------------------------------------------------
DEDUP_ENABLED = True
# Person: drop the lower-confidence box if IoU >= DEDUP_PERSON_IOU, or if it is
# mostly contained (intersection / smaller area >= DEDUP_PERSON_CONTAIN) and
# IoU >= DEDUP_PERSON_CONTAIN_MIN_IOU. A real person standing behind another
# is much smaller (IoU < 0.4) and is kept.
DEDUP_PERSON_IOU = 0.6
DEDUP_PERSON_CONTAIN = 0.85
DEDUP_PERSON_CONTAIN_MIN_IOU = 0.4
# Portable objects (backpack/handbag/suitcase) are one family for dedup.
DEDUP_OBJECT_CONTAIN = 0.7

# ---------------------------------------------------------------------------
# World state
# ---------------------------------------------------------------------------
HISTORY_SECONDS = 10.0
# An entity is "stationary" while its center stays within this fraction of
# its own box diagonal from its anchor point.
STATIONARY_RADIUS_RATIO = 0.35

# ---------------------------------------------------------------------------
# Zones (normalized 0..1 frame coordinates, polygon vertices clockwise)
# ---------------------------------------------------------------------------
# Point of the entity tested against zones: "center" or "bottom" (feet).
ZONE_ANCHOR = "center"
# Consecutive frames the anchor must be inside/outside before a zone
# enter/exit is accepted (prevents boundary jitter from spamming events).
ZONE_CONFIRM_FRAMES = 8
ZONES = [
    {
        "id": "restricted_a",
        "name": "Restricted Zone A",
        "type": "RESTRICTED",
        "points": [(0.68, 0.05), (0.97, 0.05), (0.97, 0.95), (0.68, 0.95)],
        "active": True,
    },
    {
        "id": "monitored_main",
        "name": "Monitored Area",
        "type": "MONITORED",
        "points": [(0.03, 0.05), (0.66, 0.05), (0.66, 0.95), (0.03, 0.95)],
        # Off by default: a near-full-frame zone duplicates entered/left-view
        # events. Shrink it to a real area (e.g. a doorway) and enable it.
        "active": False,
    },
]

# ---------------------------------------------------------------------------
# Temporal person events
# ---------------------------------------------------------------------------
DWELL_THRESHOLD_SECONDS = 10.0  # first DWELLING event after this long in a zone
DWELL_REPEAT_SECONDS = 15.0  # then at most one more per this interval
CROWD_THRESHOLD = 3  # crowded when people in a zone (or whole view) >= this
CROWD_CLEAR_BELOW = 3  # crowding clears when occupancy < this (set lower for more hysteresis)
CROWD_CONFIRM_SECONDS = 1.5  # condition must hold this long before firing/clearing
OCCUPANCY_CONFIRM_SECONDS = 1.0  # occupancy change must hold this long to be reported

# ---------------------------------------------------------------------------
# Person <-> object association / unattended objects
# ---------------------------------------------------------------------------
# Object is "near" a person when its distance to the person's box is below
# this many person-heights.
ASSOC_NEAR_RATIO = 0.5
# Object is "separated" when farther than this (hysteresis gap vs NEAR).
ASSOC_FAR_RATIO = 1.0
ASSOC_CANDIDATE_SECONDS = 0.5  # near this long -> CANDIDATE_ASSOCIATION
ASSOC_CONFIRM_SECONDS = 2.0  # near this long (cumulative) -> ASSOCIATED
ASSOC_GAP_TOLERANCE_SECONDS = 1.0  # brief gaps in proximity are forgiven
SEPARATION_CONFIRM_SECONDS = 1.0  # far+stationary this long -> SEPARATING
UNATTENDED_TIMEOUT_SECONDS = 5.0  # SEPARATING this long -> UNATTENDED
# Object displacement (fraction of its own box diagonal) that counts as moved.
OBJECT_MOVED_DIST_RATIO = 1.0
# How long OBJECT_MOVED stays as a dashboard alert.
OBJECT_MOVED_ALERT_SECONDS = 10.0
# Emit info "Backpack #5 near Person #2" candidate events (noisy with passers-by).
ASSOC_EMIT_CANDIDATE_EVENTS = False
# Object must be still this long before it can count as left behind.
OBJECT_STATIONARY_MIN_SECONDS = 0.5
# Tracker ID switch tolerance: a new person ID appearing next to the bag within
# this window after the associated person's track ended inherits the association.
ASSOC_ID_HANDOFF_SECONDS = 2.5
ASSOC_ID_HANDOFF_FIRST_SEEN_SLACK = 1.0
# A different person near the object within this window before it vanished
# turns "no longer visible" into OBJECT_MOVED.
OBJECT_RECENT_NEAR_SECONDS = 1.5

# ---------------------------------------------------------------------------
# Entity memory (ACTIVE -> OUT_OF_VIEW -> ARCHIVED)
# ---------------------------------------------------------------------------
# A non-associated person near an object this recently before it vanished,
# who then leaves view within LEFT_WITH_WINDOW_SECONDS, is recorded as
# "left view with" that person.
LEFT_WITH_NEAR_SECONDS = 1.5
LEFT_WITH_WINDOW_SECONDS = 5.0
SNAPSHOTS_ENABLED = True
SNAPSHOT_DIR = "data/snapshots"
SNAPSHOT_MAX_SIDE = 200  # px, longest side of the stored crop
SNAPSHOT_MIN_CONFIDENCE = 0.3
ARCHIVE_MAX_ENTITIES = 1000
RECENTLY_SEEN_LIMIT = 4

# ---------------------------------------------------------------------------
# Storage / dashboard
# ---------------------------------------------------------------------------
EVENT_DB_PATH = "data/events.sqlite3"
DASHBOARD_WIDTH = 1920
DASHBOARD_HEIGHT = 1080
TIMELINE_LENGTH = 12
DASHBOARD_DEFAULT_MODE = "demo"  # "demo" (judge-facing) or "debug"
