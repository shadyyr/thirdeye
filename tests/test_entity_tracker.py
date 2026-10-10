from __future__ import annotations

import time
import unittest
from unittest import mock

from thirdeye.core import settings
from thirdeye.core.types import Detection, TrackedEntity
from thirdeye.tracking.entity_tracker import EntityTracker

FPS = 20.0


def person(cx: float, cy: float = 400.0, conf: float = 0.9) -> Detection:
    return Detection("person", conf, (cx - 50, cy - 150, cx + 50, cy + 150))


def bag(cx: float, cy: float = 600.0, conf: float = 0.8, name: str = "backpack") -> Detection:
    return Detection(name, conf, (cx - 30, cy - 40, cx + 30, cy + 40))


class TrackerHarness:
    """Feeds frames at a fixed rate and remembers the clock."""

    def __init__(self, fps: float = FPS) -> None:
        self.tracker = EntityTracker()
        self.fps = fps
        self.frame = 0

    @property
    def now(self) -> float:
        """Timestamp the next step() will use."""
        return 100.0 + self.frame / self.fps

    def step(self, detections: list[Detection]) -> list[TrackedEntity]:
        now = self.now
        self.frame += 1
        return self.tracker.update(detections, now)

    def run_for(self, seconds: float, detections: list[Detection]) -> list[TrackedEntity]:
        out: list[TrackedEntity] = []
        for _ in range(int(round(seconds * self.fps))):
            out = self.step(detections)
        return out

    def warm_up(self, detections: list[Detection]) -> list[TrackedEntity]:
        out: list[TrackedEntity] = []
        for _ in range(settings.TRACK_MIN_HITS):
            out = self.step(detections)
        return out


def by_class(entities: list[TrackedEntity], class_name: str) -> list[TrackedEntity]:
    return [e for e in entities if e.class_name == class_name]


class EntityTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = TrackerHarness()

    def test_tentative_not_reported_before_min_hits(self) -> None:
        for _ in range(settings.TRACK_MIN_HITS - 1):
            self.assertEqual(self.h.step([person(300)]), [])
        out = self.h.step([person(300)])
        self.assertEqual([e.track_id for e in out], [1])

    def test_flicker_does_not_waste_ids(self) -> None:
        # A detection that keeps appearing for fewer than MIN_HITS frames.
        for _ in range(5):
            for _ in range(settings.TRACK_MIN_HITS - 1):
                self.h.step([person(300)])
            self.assertEqual(self.h.step([]), [])
        out = self.h.warm_up([person(800)])
        self.assertEqual(out[0].track_id, 1)

    def test_id_stable_while_walking_across(self) -> None:
        self.h.warm_up([person(80)])
        ids = set()
        for x in range(80, 1200, 25):
            out = self.h.step([person(x)])
            self.assertEqual(len(out), 1)
            ids.add(out[0].track_id)
            self.assertEqual(out[0].frames_missed, 0)
            self.assertAlmostEqual(out[0].center[0], x)
        self.assertEqual(ids, {1})

    def test_brief_miss_tolerated(self) -> None:
        self.h.warm_up([person(300)])
        for missed in range(1, 6):
            out = self.h.step([])
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].frames_missed, missed)
            self.assertEqual(out[0].bbox, person(300).bbox)  # last known box
        out = self.h.step([person(310)])
        self.assertEqual([(e.track_id, e.frames_missed) for e in out], [(1, 0)])

    def test_coasting_moving_track_reacquired(self) -> None:
        self.h.warm_up([person(100)])
        for x in range(100, 500, 20):
            self.h.step([person(x)])
        for _ in range(4):
            self.h.step([])
        out = self.h.step([person(560)])
        self.assertEqual([e.track_id for e in out], [1])

    def test_person_removed_after_max_missed_seconds(self) -> None:
        limit = settings.TRACK_MAX_MISSED_PERSON_SECONDS
        self.h.warm_up([person(300)])
        self.assertEqual(len(self.h.run_for(limit - 0.1, [])), 1)
        self.assertEqual(self.h.run_for(0.2, []), [])
        # Coming back now is a new person with a new ID.
        out = self.h.warm_up([person(300)])
        self.assertEqual([e.track_id for e in out], [2])

    def test_coasting_is_time_based(self) -> None:
        limit = settings.TRACK_MAX_MISSED_PERSON_SECONDS
        for fps in (10.0, 30.0):
            h = TrackerHarness(fps)
            h.warm_up([person(300)])
            last_detected = h.now - 1.0 / fps
            gone_at = None
            for _ in range(int(fps * (limit + 1.0))):
                t = h.now
                if not h.step([]):
                    gone_at = t
                    break
            self.assertIsNotNone(gone_at)
            elapsed = gone_at - last_detected
            self.assertGreater(elapsed, limit, fps)
            self.assertLessEqual(elapsed, limit + 1.0 / fps + 1e-6, fps)

    def test_object_survives_longer_than_person(self) -> None:
        p_limit = settings.TRACK_MAX_MISSED_PERSON_SECONDS
        o_limit = settings.TRACK_MAX_MISSED_OBJECT_SECONDS
        self.h.warm_up([person(300), bag(800)])
        out = self.h.run_for(p_limit + 0.1, [])
        self.assertEqual([e.class_name for e in out], ["backpack"])
        out = self.h.run_for(o_limit - p_limit - 0.2, [])
        self.assertEqual(len(out), 1)
        self.assertEqual(self.h.run_for(0.2, []), [])

    def test_two_people_crossing_keep_ids(self) -> None:
        a, b = 200.0, 1000.0
        out = self.h.warm_up([person(a, 380), person(b, 420)])
        ids = {round(e.center[0]): e.track_id for e in out}
        id_a, id_b = ids[200], ids[1000]
        for _ in range(40):
            a += 20
            b -= 20
            out = self.h.step([person(b, 420), person(a, 380)])
            self.assertEqual(len(out), 2)
            got = {e.track_id: e.center[0] for e in out}
            self.assertAlmostEqual(got[id_a], a)
            self.assertAlmostEqual(got[id_b], b)
        self.assertGreater(a, b)  # they really crossed
        self.assertEqual(self.h.tracker.merges, [])

    def test_slow_crossing_people_not_merged(self) -> None:
        # Relative speed 200 px/s: boxes overlap heavily for well under
        # TRACK_MERGE_SECONDS, so the two people must stay distinct.
        a, b = 400.0, 600.0
        self.h.warm_up([person(a, 390), person(b, 410)])
        seen: set[int] = set()
        for _ in range(40):
            a += 5
            b -= 5
            out = self.h.step([person(a, 390), person(b, 410)])
            self.assertEqual(len(out), 2)
            seen.update(e.track_id for e in out)
        self.assertGreater(a, b)
        self.assertEqual(seen, {1, 2})
        self.assertEqual(self.h.tracker.merges, [])

    def test_side_by_side_people_stay_distinct(self) -> None:
        # Partially overlapping (IoU ~0.29) for several seconds.
        self.h.warm_up([person(300), person(355)])
        out = self.h.run_for(5.0, [person(300), person(355)])
        self.assertEqual([e.track_id for e in out], [1, 2])
        self.assertEqual(self.h.tracker.merges, [])

    def test_class_separation(self) -> None:
        self.h.warm_up([person(300)])
        # A backpack exactly where the person was must not continue the person.
        out = self.h.step([Detection("backpack", 0.9, person(300).bbox)])
        self.assertEqual([(e.class_name, e.frames_missed) for e in out], [("person", 1)])
        out = self.h.step([Detection("backpack", 0.9, person(300).bbox)])
        out = self.h.step([Detection("backpack", 0.9, person(300).bbox)])
        self.assertEqual({e.class_name: e.track_id for e in out}, {"person": 1, "backpack": 2})

    def test_portable_label_flicker_keeps_one_id(self) -> None:
        labels = ["backpack", "handbag", "backpack", "suitcase", "backpack", "handbag"]
        out = []
        ids: set[int] = set()
        for i in range(60):
            name = labels[i % len(labels)]
            out = self.h.step([bag(700 + i % 3, conf=0.45, name=name)])
            ids.update(e.track_id for e in out)
            if out:
                self.assertEqual(len(out), 1)
        self.assertEqual(ids, {1})
        self.assertEqual(out[0].class_name, "backpack")  # majority label

    def test_single_odd_label_does_not_flip_class(self) -> None:
        self.h.run_for(1.0, [bag(700, name="backpack")])
        out = self.h.step([bag(700, conf=0.9, name="suitcase")])
        self.assertEqual([(e.track_id, e.class_name) for e in out], [(1, "backpack")])

    def test_label_switch_keeps_id_and_class_follows_votes(self) -> None:
        self.h.run_for(0.5, [bag(700, name="backpack")])
        out = self.h.run_for(3.0, [bag(700, name="handbag")])
        self.assertEqual([(e.track_id, e.class_name) for e in out], [(1, "handbag")])

    def test_low_confidence_extends_but_does_not_create(self) -> None:
        low = settings.TRACK_NEW_CONFIDENCE["person"] - 0.1
        for _ in range(10):
            self.assertEqual(self.h.step([person(300, conf=low)]), [])
        self.h.warm_up([person(800)])
        for step, x in enumerate(range(800, 900, 10)):
            out = self.h.step([person(x, conf=low)])
            self.assertEqual([(e.track_id, e.frames_missed) for e in out], [(1, 0)])
            self.assertAlmostEqual(out[0].center[0], x)
            if step == 0:
                # Reported confidence is a recent max, so one weak frame does not drop it.
                self.assertAlmostEqual(out[0].confidence, 0.9)

    def test_low_confidence_extension_window(self) -> None:
        window = settings.TRACK_LOW_CONF_EXTEND_SECONDS
        low = settings.TRACK_NEW_CONFIDENCE["person"] - 0.1
        self.h.warm_up([person(300)])
        # Undetected for longer than the window: a weak box cannot re-acquire it.
        self.h.run_for(window + 0.1, [])
        out = self.h.step([person(300, conf=low)])
        self.assertEqual(len(out), 1)
        self.assertGreater(out[0].frames_missed, 0)
        # A confident box still can.
        out = self.h.step([person(300)])
        self.assertEqual([(e.track_id, e.frames_missed) for e in out], [(1, 0)])
        # Within the window, a weak box extends it.
        self.h.run_for(max(window - 0.15, 0.05), [])
        out = self.h.step([person(300, conf=low)])
        self.assertEqual([(e.track_id, e.frames_missed) for e in out], [(1, 0)])

    def test_low_confidence_cannot_keep_ghost_alive_after_gap(self) -> None:
        low = settings.TRACK_NEW_CONFIDENCE["person"] - 0.1
        self.h.warm_up([person(300)])
        self.h.run_for(settings.TRACK_LOW_CONF_EXTEND_SECONDS + 0.1, [])
        out = self.h.run_for(settings.TRACK_MAX_MISSED_PERSON_SECONDS, [person(300, conf=low)])
        self.assertEqual(out, [])

    def test_low_confidence_does_not_steal_from_high(self) -> None:
        low = settings.TRACK_NEW_CONFIDENCE["person"] - 0.1
        self.h.warm_up([person(300)])
        out = self.h.step([person(330, conf=low), person(305)])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].center[0], 305)

    def test_duplicate_person_tracks_merge_into_older(self) -> None:
        full = Detection("person", 0.8, (100, 100, 200, 400))
        upper = Detection("person", 0.6, (100, 100, 200, 300))  # nested upper body
        out = self.h.warm_up([full, upper])
        self.assertEqual([e.track_id for e in out], [1, 2])
        out = self.h.run_for(settings.TRACK_MERGE_SECONDS + 0.1, [full, upper])
        self.assertIn(1, [e.track_id for e in out])
        self.assertNotIn(2, [e.track_id for e in out])
        self.assertEqual([(k, m) for _, k, m in self.h.tracker.merges[:1]], [(1, 2)])

    def test_merge_waits_for_merge_seconds(self) -> None:
        full = Detection("person", 0.8, (100, 100, 200, 400))
        upper = Detection("person", 0.6, (100, 100, 200, 300))
        self.h.warm_up([full, upper])
        out = self.h.run_for(settings.TRACK_MERGE_SECONDS - 0.15, [full, upper])
        self.assertEqual([e.track_id for e in out], [1, 2])
        self.assertEqual(self.h.tracker.merges, [])

    def test_stale_older_track_absorbs_new_duplicate(self) -> None:
        self.h.warm_up([bag(700)])  # Backpack #1
        self.h.run_for(0.5, [])  # coasting
        moved = bag(706, 604, name="handbag")
        # Force the new box not to match the stale track (as happened live).
        with mock.patch.object(settings, "TRACK_MIN_IOU", 1.1), \
                mock.patch.object(settings, "TRACK_MAX_CENTER_DIST_RATIO", 0.001):
            out = self.h.warm_up([moved])
            self.assertEqual(sorted(e.track_id for e in out), [1, 2])
            out = self.h.run_for(settings.TRACK_MERGE_SECONDS + 0.1, [moved])
        self.assertEqual([(e.track_id, e.frames_missed) for e in out], [(1, 0)])
        self.assertEqual(out[0].bbox, moved.bbox)  # took over the fresh box
        self.assertEqual(self.h.tracker.merges[0][1:], (1, 2))
        # And keeps following the bag afterwards.
        out = self.h.run_for(0.5, [moved])
        self.assertEqual([e.track_id for e in out], [1])

    def test_departed_person_does_not_absorb_newcomer(self) -> None:
        # Person #1 leaves; their stale coasting box overlaps a different person
        # arriving at the same spot. The newcomer must keep a new ID.
        self.h.warm_up([person(300)])
        self.h.run_for(0.1, [])  # #1 coasting, no longer detected
        newcomer = Detection("person", 0.8, (255, 260, 345, 470))
        with mock.patch.object(settings, "TRACK_MIN_IOU", 1.1), \
                mock.patch.object(settings, "TRACK_MAX_CENTER_DIST_RATIO", 0.001):
            out = self.h.warm_up([newcomer])
            out = self.h.run_for(settings.TRACK_MERGE_SECONDS + 0.1, [newcomer])
        self.assertIn(2, [e.track_id for e in out])
        self.assertEqual(self.h.tracker.merges, [])

    def test_object_revive_keeps_old_id(self) -> None:
        self.h.warm_up([person(200)])  # Person #1
        self.h.warm_up([person(200), bag(700)])  # Backpack #2
        # Bag disappears long enough to be removed (person walks off too).
        out = self.h.run_for(settings.TRACK_MAX_MISSED_OBJECT_SECONDS + 0.3, [])
        self.assertEqual(out, [])
        # Bag is detected again, slightly shifted, within the revive window.
        out = self.h.warm_up([bag(712, 605)])
        self.assertEqual([(e.class_name, e.track_id) for e in out], [("backpack", 2)])
        # A new person afterwards gets the next fresh ID.
        out = self.h.warm_up([bag(712, 605), person(200)])
        self.assertEqual({e.class_name: e.track_id for e in out}, {"backpack": 2, "person": 3})

    def test_object_revive_rules(self) -> None:
        self.h.warm_up([bag(700)])  # Backpack #1
        self.h.run_for(settings.TRACK_MAX_MISSED_OBJECT_SECONDS + 0.1, [])
        # Far away -> new ID; a person at the same spot never takes a bag ID.
        out = self.h.warm_up([bag(200), Detection("person", 0.9, bag(700).bbox)])
        self.assertEqual(sorted((e.class_name, e.track_id) for e in out),
                         [("backpack", 2), ("person", 3)])
        # Same spot, other portable label -> old ID (same family), only once.
        out = self.h.warm_up([bag(200), Detection("person", 0.9, bag(700).bbox),
                              bag(700, 560, name="handbag")])
        self.assertIn(("handbag", 1), {(e.class_name, e.track_id) for e in out})

    def test_object_revive_expires(self) -> None:
        self.h.warm_up([bag(700)])
        self.h.run_for(settings.TRACK_MAX_MISSED_OBJECT_SECONDS + 0.1, [])
        self.h.run_for(settings.TRACK_OBJECT_REVIVE_SECONDS + 0.3, [])
        out = self.h.warm_up([bag(700)])
        self.assertEqual([e.track_id for e in out], [2])

    def test_person_is_never_revived(self) -> None:
        self.h.warm_up([person(300)])
        self.h.run_for(settings.TRACK_MAX_MISSED_PERSON_SECONDS + 0.1, [])
        out = self.h.warm_up([person(300)])
        self.assertEqual([e.track_id for e in out], [2])

    def test_global_id_counter_shared_across_classes(self) -> None:
        self.h.warm_up([person(200)])
        self.h.warm_up([person(200), bag(600)])
        out = self.h.warm_up([person(200), bag(600), person(1000), bag(900, name="handbag")])
        labels = sorted((e.track_id, e.class_name) for e in out)
        self.assertEqual(labels, [(1, "person"), (2, "backpack"), (3, "person"), (4, "handbag")])

    def test_reset(self) -> None:
        self.h.warm_up([person(300)])
        self.h.tracker.reset()
        self.assertEqual(self.h.step([]), [])
        self.assertEqual(self.h.tracker.merges, [])
        out = self.h.warm_up([person(300)])
        self.assertEqual([e.track_id for e in out], [1])

    def test_output_fields(self) -> None:
        out = self.h.warm_up([person(300)])
        e = out[0]
        self.assertEqual(e.bbox, person(300).bbox)
        self.assertEqual(e.center, (300.0, 400.0))
        self.assertAlmostEqual(e.timestamp, 100.0 + (settings.TRACK_MIN_HITS - 1) / FPS)

    def test_speed(self) -> None:
        dets = [person(100 + 110 * i) for i in range(6)] + [bag(150 + 200 * i) for i in range(4)]
        self.h.warm_up(dets)
        frames = 300
        start = time.perf_counter()
        for k in range(frames):
            shifted = [Detection(d.class_name, d.confidence,
                                 (d.bbox[0] + k % 5, d.bbox[1], d.bbox[2] + k % 5, d.bbox[3]))
                       for d in dets]
            self.h.step(shifted)
        per_frame_ms = (time.perf_counter() - start) * 1000 / frames
        # Target is < 1 ms on the Jetson; generous bound keeps CI stable.
        self.assertLess(per_frame_ms, 5.0)


if __name__ == "__main__":
    unittest.main()
