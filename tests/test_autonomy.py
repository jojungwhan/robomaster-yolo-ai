import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from autonomy import (
    AutonomousPlanner,
    DryRunBackend,
    MissionLogger,
    MotionCommand,
    SafetyMotionController,
)
from lego_vision import (
    HELP_SIGNAL_LABEL,
    ConsecutiveDetectionGate,
    detect_aruco_lego_markers,
    detect_colored_lego_candidates,
    detect_red_cross_signal,
    generate_marker,
)


class MotionSafetyTests(unittest.TestCase):
    def make_controller(self, backend=None, motion_requested=True):
        self.temp_dir = tempfile.TemporaryDirectory()
        backend = backend or DryRunBackend(2000)
        controller = SafetyMotionController(
            backend,
            motion_requested=motion_requested,
            logger=MissionLogger(Path(self.temp_dir.name) / "events.jsonl"),
        )
        controller.connect()
        self.addCleanup(controller.close)
        self.addCleanup(self.temp_dir.cleanup)
        return controller, backend

    def test_motion_flag_is_required(self):
        controller, _ = self.make_controller(motion_requested=False)
        self.assertFalse(controller.arm())
        self.assertIn("--enable-motion", controller.last_reason)

    def test_four_live_tof_channels_are_required(self):
        backend = DryRunBackend(2000)
        backend.set_distances((2000,))
        controller, _ = self.make_controller(backend=backend)
        self.assertFalse(controller.arm())
        self.assertIn("ToF", controller.last_reason)

    def test_wall_blocks_forward_motion(self):
        controller, backend = self.make_controller()
        self.assertTrue(controller.arm())
        backend.set_distances((500, 2000, 2000, 2000))
        result = controller.apply(
            MotionCommand(forward_mps=0.10, reason="forward"),
            [],
            (720, 1280, 3),
        )
        self.assertEqual(result.forward_mps, 0)
        self.assertIn("blocked", controller.last_reason)

    def test_large_person_in_path_stops_motion(self):
        controller, _ = self.make_controller()
        self.assertTrue(controller.arm())
        person = {
            "label": "person",
            "confidence": 0.9,
            "box": (420, 200, 860, 680),
        }
        result = controller.apply(
            MotionCommand(forward_mps=0.10, reason="approach"),
            [person],
            (720, 1280, 3),
        )
        self.assertEqual(result.forward_mps, 0)
        self.assertIn("person", controller.last_reason)

    def test_impact_latches_emergency_stop(self):
        controller, backend = self.make_controller()
        self.assertTrue(controller.arm())
        backend.impact = True
        controller.apply(MotionCommand(forward_mps=0.10), [], (720, 1280, 3))
        self.assertTrue(controller.emergency_latched)
        self.assertFalse(controller.armed)

    def test_planner_turns_toward_open_side(self):
        planner = AutonomousPlanner()
        command = planner.plan(
            "exploration",
            [],
            "person",
            (720, 1280, 3),
            {"front": 500, "left": 1600, "right": 800, "rear": 1200},
        )
        self.assertGreater(command.yaw_dps, 0)
        self.assertEqual(command.forward_mps, 0)


class LegoVisionTests(unittest.TestCase):
    def test_aruco_marker_detection(self):
        marker = generate_marker(7, 400)
        canvas = np.full((600, 600), 255, dtype=np.uint8)
        canvas[100:500, 100:500] = marker
        frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        labels = {
            item["label"] for item in detect_aruco_lego_markers(frame)
        }
        self.assertIn("lego_marker_7", labels)

    def test_colored_rectangle_candidate(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(frame, (400, 250), (700, 420), (255, 0, 0), -1)
        labels = {
            item["label"] for item in detect_colored_lego_candidates(frame)
        }
        self.assertIn("lego_blue", labels)

    def test_red_cross_is_a_help_signal(self):
        frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (600, 190), (680, 530), (0, 0, 220), -1)
        cv2.rectangle(frame, (470, 320), (810, 400), (0, 0, 220), -1)
        detections = detect_red_cross_signal(frame)
        self.assertIn(HELP_SIGNAL_LABEL, {item["label"] for item in detections})
        self.assertEqual(detections[0]["signal"], "help_needed")

    def test_red_square_is_not_a_help_signal(self):
        frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (480, 200), (800, 520), (0, 0, 220), -1)
        self.assertEqual(detect_red_cross_signal(frame), [])

    def test_help_signal_requires_consecutive_frames(self):
        gate = ConsecutiveDetectionGate(required_frames=3, release_frames=2)
        self.assertEqual(gate.update(True), (False, False))
        self.assertEqual(gate.update(True), (False, False))
        self.assertEqual(gate.update(True), (True, True))
        self.assertEqual(gate.update(True), (True, False))
        self.assertEqual(gate.update(False), (True, False))
        self.assertEqual(gate.update(False), (False, False))


if __name__ == "__main__":
    unittest.main()
