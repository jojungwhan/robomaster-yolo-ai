from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from navigation import (
    GimbalTracker,
    MissionMap,
    MissionNavigator,
    NavigationState,
    TargetTracker,
)
from scenario_profiles import ScenarioCatalog, ScenarioProfile


ROOT = Path(__file__).resolve().parents[1]


def detection(label="person", box=(500, 260, 700, 470), confidence=0.9):
    return {"label": label, "box": box, "confidence": confidence}


def motion_status(mode="ARMED", position=(0.0, 0.0, 0.0), front=2000):
    return {
        "mode": mode,
        "position": position,
        "ranges": {
            "front": front,
            "left": 2000,
            "right": 2000,
            "rear": 2000,
        },
    }


class ScenarioProfileTests(unittest.TestCase):
    def test_default_catalog_contains_classroom_scenarios(self):
        catalog = ScenarioCatalog.load(ROOT / "scenarios.json")
        self.assertGreaterEqual(len(catalog), 10)
        self.assertEqual(catalog.get("rescue").navigation_policy, "rescue")
        self.assertTrue(catalog.get("lego").use_lego_vision)
        self.assertEqual(catalog.key_map[ord("1")], "exploration")

    def test_duplicate_scenario_shortcuts_are_rejected(self):
        catalog = ScenarioCatalog.load(ROOT / "scenarios.json")
        first = catalog.get("exploration")
        second = catalog.get("rescue")
        duplicate = second.__class__(**{**second.__dict__, "key": first.key})
        with self.assertRaisesRegex(ValueError, "shortcuts"):
            ScenarioCatalog((first, duplicate))

    def test_scenario_target_allowlist_is_enforced(self):
        catalog = ScenarioCatalog.load(ROOT / "scenarios.json")
        follow = catalog.get("follow")
        self.assertTrue(follow.accepts_target("person"))
        self.assertFalse(follow.accepts_target("tv"))

    def test_completion_condition_must_match_navigation_policy(self):
        delivery = ScenarioCatalog.load(ROOT / "scenarios.json").get("delivery")
        with self.assertRaisesRegex(ValueError, "requires return_home"):
            ScenarioProfile.from_dict(
                {**delivery.__dict__, "completion": "home_reached"}
            )


class TargetTrackerTests(unittest.TestCase):
    def test_requires_multiple_frames_and_holds_lock_across_short_miss(self):
        tracker = TargetTracker(acquire_frames=2)
        item = detection()
        first = tracker.update([item], "person", now=1.0)
        second = tracker.update([item], "person", now=1.1)
        missed = tracker.update([], "person", now=1.5)

        self.assertFalse(first.locked)
        self.assertTrue(second.locked)
        self.assertTrue(missed.locked)
        self.assertTrue(missed.temporarily_lost)
        self.assertIsNone(missed.detection)

    def test_does_not_switch_to_far_same_label_distractor(self):
        tracker = TargetTracker(acquire_frames=1)
        original = detection(box=(100, 100, 300, 500), confidence=0.8)
        distractor = detection(box=(900, 100, 1150, 600), confidence=0.99)
        tracker.update([original], "person", now=1.0)
        result = tracker.update([distractor], "person", now=1.1)

        self.assertIsNone(result.detection)
        self.assertTrue(result.temporarily_lost)
        self.assertEqual(tracker.last_box, original["box"])

    def test_cached_observation_does_not_fake_multiframe_confirmation(self):
        tracker = TargetTracker(acquire_frames=2)
        cached = {
            **detection(),
            "observation_id": "yolo:1",
        }
        first = tracker.update([cached], "person", now=1.0)
        repeated = tracker.update([cached], "person", now=1.1)
        fresh = tracker.update(
            [{**cached, "observation_id": "yolo:2"}],
            "person",
            now=1.2,
        )
        self.assertFalse(first.locked)
        self.assertFalse(repeated.locked)
        self.assertTrue(fresh.locked)
        self.assertTrue(fresh.just_locked)


class MissionMapTests(unittest.TestCase):
    def test_map_records_home_waypoints_and_round_trips_json(self):
        mission_map = MissionMap()
        mission_map.record((1.0, 2.0, 0.0), {"front": 1000})
        waypoint = mission_map.add_waypoint("Alpha", (2.0, 3.0, 90.0))
        self.assertEqual(mission_map.home, (1.0, 2.0))
        self.assertEqual((waypoint.x, waypoint.y), (2.0, 3.0))
        self.assertTrue(mission_map.obstacles)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "map.json"
            mission_map.save(path)
            restored = MissionMap.load(path)
        self.assertEqual(restored.home, mission_map.home)
        self.assertEqual(restored.waypoints[0].name, "Alpha")


class MissionNavigatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = ScenarioCatalog.load(ROOT / "scenarios.json")

    def test_target_navigation_confirms_then_approaches(self):
        navigator = MissionNavigator()
        profile = self.catalog.get("target")
        item = detection()
        first = navigator.plan(
            profile,
            [item],
            "person",
            (720, 1280, 3),
            motion_status(),
            now=1.0,
        )
        second = navigator.plan(
            profile,
            [item],
            "person",
            (720, 1280, 3),
            motion_status(),
            now=1.1,
        )
        third = navigator.plan(
            profile,
            [item],
            "person",
            (720, 1280, 3),
            motion_status(),
            now=1.2,
        )
        self.assertEqual(first.state, NavigationState.ACQUIRING)
        self.assertEqual(first.command.forward_mps, 0)
        self.assertEqual(second.state, NavigationState.TARGET_LOCKED)
        self.assertEqual(second.command.forward_mps, 0)
        self.assertEqual(third.state, NavigationState.APPROACH)
        self.assertGreater(third.command.forward_mps, 0)

    def test_lost_target_holds_position_before_scanning(self):
        navigator = MissionNavigator()
        profile = self.catalog.get("target")
        item = detection()
        navigator.plan(profile, [item], "person", (720, 1280, 3), motion_status(), now=1.0)
        navigator.plan(profile, [item], "person", (720, 1280, 3), motion_status(), now=1.1)
        result = navigator.plan(profile, [], "person", (720, 1280, 3), motion_status(), now=1.5)
        self.assertEqual(result.state, NavigationState.RECOVERY)
        self.assertEqual(result.command.forward_mps, 0)
        self.assertEqual(result.command.yaw_dps, 0)

    def test_pause_state_takes_priority_over_active_conversation(self):
        navigator = MissionNavigator()
        result = navigator.plan(
            self.catalog.get("target"),
            [detection()],
            "person",
            (720, 1280, 3),
            motion_status(),
            paused=True,
            interaction_active=True,
        )
        self.assertEqual(result.state, NavigationState.PAUSED)
        self.assertEqual(result.command.forward_mps, 0)

    def test_close_target_enters_interaction_state(self):
        navigator = MissionNavigator()
        profile = self.catalog.get("target")
        close = detection(box=(500, 100, 780, 650))
        navigator.plan(profile, [close], "person", (720, 1280, 3), motion_status(), now=1.0)
        navigator.plan(profile, [close], "person", (720, 1280, 3), motion_status(), now=1.1)
        result = navigator.plan(profile, [close], "person", (720, 1280, 3), motion_status(), now=1.2)
        self.assertEqual(result.state, NavigationState.INTERACT)
        self.assertEqual(result.command.forward_mps, 0)

    def test_waypoint_policy_drives_toward_selected_waypoint(self):
        mission_map = MissionMap()
        mission_map.record((0.0, 0.0, 0.0), {})
        mission_map.add_waypoint("Goal", (1.0, 0.0, 0.0))
        navigator = MissionNavigator(mission_map=mission_map)
        result = navigator.plan(
            self.catalog.get("delivery"),
            [],
            "person",
            (720, 1280, 3),
            motion_status(position=(0.0, 0.0, 0.0)),
        )
        self.assertEqual(result.state, NavigationState.WAYPOINT)
        self.assertGreater(result.command.forward_mps, 0)
        self.assertAlmostEqual(result.distance_to_goal_m, 1.0)

    def test_waypoint_obstacle_enters_recovery_before_advancing(self):
        mission_map = MissionMap()
        mission_map.record((0.0, 0.0, 0.0), {})
        mission_map.add_waypoint("Goal", (1.0, 0.0, 0.0))
        navigator = MissionNavigator(mission_map=mission_map)
        result = navigator.plan(
            self.catalog.get("delivery"),
            [],
            "person",
            (720, 1280, 3),
            motion_status(position=(0.0, 0.0, 0.0), front=500),
        )
        self.assertEqual(result.state, NavigationState.RECOVERY)
        self.assertEqual(result.command.forward_mps, 0)
        self.assertNotEqual(result.command.yaw_dps, 0)

    def test_scan_alternates_direction_and_times_out_stopped(self):
        navigator = MissionNavigator(
            scan_timeout_seconds=3.0,
            scan_sweep_seconds=1.0,
        )
        profile = self.catalog.get("target")
        first = navigator.plan(
            profile, [], "person", (720, 1280, 3), motion_status(), now=1.0
        )
        second = navigator.plan(
            profile, [], "person", (720, 1280, 3), motion_status(), now=2.1
        )
        timed_out = navigator.plan(
            profile, [], "person", (720, 1280, 3), motion_status(), now=4.0
        )
        self.assertEqual(first.state, NavigationState.SCAN)
        self.assertEqual(second.state, NavigationState.SCAN)
        self.assertLess(first.command.yaw_dps * second.command.yaw_dps, 0)
        self.assertEqual(timed_out.state, NavigationState.RECOVERY)
        self.assertEqual(timed_out.command, timed_out.command.__class__(reason=timed_out.reason))

    def test_goal_proximity_reports_complete_with_zero_motion(self):
        mission_map = MissionMap()
        mission_map.record((0.0, 0.0, 0.0), {})
        mission_map.add_waypoint("Goal", (0.1, 0.0, 0.0))
        navigator = MissionNavigator(mission_map=mission_map)
        result = navigator.plan(
            self.catalog.get("delivery"),
            [],
            "person",
            (720, 1280, 3),
            motion_status(),
        )
        self.assertEqual(result.state, NavigationState.COMPLETE)
        self.assertEqual(result.command.forward_mps, 0)
        self.assertEqual(result.command.yaw_dps, 0)

    def test_operator_completion_waits_stopped_at_waypoint(self):
        mission_map = MissionMap()
        mission_map.record((0.0, 0.0, 0.0), {})
        mission_map.add_waypoint("Goal", (0.1, 0.0, 0.0))
        profile = replace(self.catalog.get("delivery"), completion="operator")
        result = MissionNavigator(mission_map=mission_map).plan(
            profile,
            [],
            "person",
            (720, 1280, 3),
            motion_status(),
        )
        self.assertEqual(result.state, NavigationState.INTERACT)
        self.assertEqual(result.command.forward_mps, 0)


class GimbalTrackerTests(unittest.TestCase):
    def test_target_offset_generates_bounded_pitch_and_yaw(self):
        tracker = GimbalTracker(max_speed_dps=20.0)
        command = tracker.plan(
            detection(box=(10, 10, 110, 110)),
            (720, 1280, 3),
        )
        self.assertGreater(command.yaw_dps, 0)
        self.assertGreater(command.pitch_dps, 0)
        self.assertLessEqual(abs(command.yaw_dps), 20.0)
        self.assertLessEqual(abs(command.pitch_dps), 20.0)

    def test_centered_target_holds_gimbal(self):
        tracker = GimbalTracker()
        command = tracker.plan(
            detection(box=(590, 310, 690, 410)),
            (720, 1280, 3),
        )
        self.assertEqual(command.pitch_dps, 0)
        self.assertEqual(command.yaw_dps, 0)
        self.assertEqual(command.reason, "target centered")


if __name__ == "__main__":
    unittest.main()
