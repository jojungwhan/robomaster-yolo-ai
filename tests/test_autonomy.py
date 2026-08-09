import json
import tempfile
from pathlib import Path
import threading
import time
import unittest

import cv2
import numpy as np

from autonomy import (
    AutonomousPlanner,
    DryRunBackend,
    MissionLogger,
    MotionCommand,
    RoboMasterBackend,
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

    def test_emergency_reset_is_rejected_when_not_latched(self):
        controller, _ = self.make_controller()
        self.assertFalse(controller.reset_emergency_stop())
        self.assertFalse(controller.emergency_latched)

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

    def test_planner_stops_when_target_is_visually_close(self):
        planner = AutonomousPlanner()
        target = {
            "label": "lego_marker_7",
            "confidence": 0.95,
            "box": (500, 100, 780, 650),
        }
        command = planner.plan(
            "lego",
            [target],
            "lego_marker_7",
            (720, 1280, 3),
            {"front": 2000, "left": 2000, "right": 2000, "rear": 2000},
        )
        self.assertEqual(command.forward_mps, 0)
        self.assertEqual(command.yaw_dps, 0)
        self.assertIn("close", command.reason)

    def test_duplicate_tof_directions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            SafetyMotionController(
                DryRunBackend(),
                tof_layout=("front", "front", "right", "rear"),
            )

    def test_reverse_motion_checks_rear_clearance(self):
        controller, backend = self.make_controller()
        self.assertTrue(controller.arm())
        backend.set_distances((2000, 2000, 2000, 400))
        command = controller.apply(
            MotionCommand(forward_mps=-0.1, reason="reverse"),
            [],
            (720, 1280, 3),
        )
        self.assertEqual(command.forward_mps, 0)
        self.assertIn("reverse blocked", controller.last_reason)

    def test_emergency_state_latches_even_when_sdk_stop_fails(self):
        class FailingStopBackend(DryRunBackend):
            def stop(self):
                raise RuntimeError("SDK offline")

            def close(self):
                self.connected = False

        controller, _ = self.make_controller(backend=FailingStopBackend())
        self.assertTrue(controller.arm())
        controller.emergency_stop("test emergency")
        self.assertTrue(controller.emergency_latched)
        self.assertFalse(controller.armed)
        self.assertIn("stop command failed", controller.last_reason)

    def test_latched_impact_is_logged_only_once(self):
        controller, backend = self.make_controller()
        self.assertTrue(controller.arm())
        backend.impact = True
        controller.apply(MotionCommand(forward_mps=0.1), [], (720, 1280, 3))
        controller.apply(MotionCommand(forward_mps=0.1), [], (720, 1280, 3))

        log_path = Path(self.temp_dir.name) / "events.jsonl"
        events = [json.loads(line)["event"] for line in log_path.read_text().splitlines()]
        self.assertEqual(events.count("emergency_stop"), 1)

    def test_async_emergency_stop_wins_over_inflight_apply(self):
        class SnapshotBarrierBackend(DryRunBackend):
            def __init__(self):
                super().__init__(2000)
                self.block_snapshot = False
                self.snapshot_started = threading.Event()
                self.release_snapshot = threading.Event()

            def snapshot(self):
                if self.block_snapshot:
                    self.snapshot_started.set()
                    self.release_snapshot.wait(timeout=1)
                return super().snapshot()

        backend = SnapshotBarrierBackend()
        controller, _ = self.make_controller(backend=backend)
        self.assertTrue(controller.arm())
        backend.block_snapshot = True

        apply_thread = threading.Thread(
            target=controller.apply,
            args=(MotionCommand(forward_mps=0.1), [], (720, 1280, 3)),
        )
        apply_thread.start()
        self.assertTrue(backend.snapshot_started.wait(timeout=1))

        stop_thread = threading.Thread(
            target=controller.emergency_stop,
            args=("asynchronous stop",),
        )
        stop_thread.start()
        time.sleep(0.02)
        backend.release_snapshot.set()
        apply_thread.join(timeout=1)
        stop_thread.join(timeout=1)

        self.assertFalse(apply_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        self.assertTrue(controller.emergency_latched)
        self.assertEqual(backend.last_command.forward_mps, 0)

    def test_hardware_stop_attempts_gimbal_when_chassis_stop_fails(self):
        class FailingChassis:
            def drive_speed(self, **_kwargs):
                raise RuntimeError("chassis offline")

            def unsub_position(self):
                pass

            def unsub_status(self):
                pass

        class RecordingGimbal:
            def __init__(self):
                self.commands = []

            def drive_speed(self, **kwargs):
                self.commands.append(kwargs)

        backend = RoboMasterBackend()
        backend.chassis = FailingChassis()
        backend.gimbal = RecordingGimbal()

        with self.assertRaisesRegex(RuntimeError, "chassis offline"):
            backend.stop()
        self.assertEqual(
            backend.gimbal.commands,
            [{"pitch_speed": 0, "yaw_speed": 0}],
        )

        class RecordingRobot:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        backend.robot = RecordingRobot()
        with self.assertRaisesRegex(RuntimeError, "chassis offline"):
            backend.close()
        self.assertTrue(backend.robot.closed)
        self.assertFalse(backend.connected)

    def test_hardware_impact_callback_fires_once_until_reset(self):
        backend = RoboMasterBackend()
        callbacks = []
        backend.set_emergency_callback(callbacks.append)
        impact_status = (False,) * 6 + (True, False, False)
        backend._on_status(impact_status)
        backend._on_status(impact_status)
        self.assertEqual(callbacks, ["hardware impact detected"])
        backend.clear_impact()
        backend._on_status(impact_status)
        self.assertEqual(len(callbacks), 2)

    def test_async_emergency_stop_wins_over_inflight_gimbal_command(self):
        class GimbalBarrierBackend(DryRunBackend):
            def __init__(self):
                super().__init__(2000)
                self.gimbal_started = threading.Event()
                self.release_gimbal = threading.Event()

            def drive_gimbal(self, pitch_speed, yaw_speed):
                self.gimbal_started.set()
                self.release_gimbal.wait(timeout=1)
                super().drive_gimbal(pitch_speed, yaw_speed)

        backend = GimbalBarrierBackend()
        controller, _ = self.make_controller(backend=backend)
        self.assertTrue(controller.arm())
        gimbal_thread = threading.Thread(
            target=controller.apply_gimbal,
            args=(20, -20),
        )
        gimbal_thread.start()
        self.assertTrue(backend.gimbal_started.wait(timeout=1))
        stop_thread = threading.Thread(
            target=controller.emergency_stop,
            args=("asynchronous gimbal stop",),
        )
        stop_thread.start()
        backend.release_gimbal.set()
        gimbal_thread.join(timeout=1)
        stop_thread.join(timeout=1)

        self.assertTrue(controller.emergency_latched)
        self.assertEqual(backend.last_gimbal_command, (0.0, 0.0))

    def test_gimbal_failure_latches_emergency_state(self):
        class FailingGimbalBackend(DryRunBackend):
            def drive_gimbal(self, _pitch_speed, _yaw_speed):
                raise RuntimeError("gimbal offline")

        controller, _ = self.make_controller(backend=FailingGimbalBackend())
        self.assertTrue(controller.arm())
        self.assertFalse(controller.apply_gimbal(10, 10))
        self.assertTrue(controller.emergency_latched)
        self.assertFalse(controller.armed)


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
