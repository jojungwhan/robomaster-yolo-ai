from pathlib import Path
import unittest

import cv2
import numpy as np

from dashboard import ClassroomDashboard
from navigation import MissionMap, NavigationDecision, NavigationState
from autonomy import MotionCommand
from scenario_profiles import ScenarioCatalog


ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = ScenarioCatalog.load(ROOT / "scenarios.json")

    def make_render_args(self, dashboard, profile_id="exploration"):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = [
            {
                "label": "person",
                "box": (400, 180, 700, 650),
                "confidence": 0.91,
            }
        ]
        state = {
            "mission_paused": False,
            "student_caption": "I need help",
            "robot_caption": "How can I assist?",
        }
        motion = {
            "mode": "DISARMED",
            "reason": "test",
            "sensor_ready": True,
            "position": (0.0, 0.0, 0.0),
            "ranges": {"front": 2000, "left": 1800, "right": 1600, "rear": 1900},
        }
        navigation = NavigationDecision(
            NavigationState.DISARMED,
            MotionCommand(),
            None,
            "test",
        )
        return (
            frame,
            detections,
            state,
            self.catalog.get(profile_id),
            tuple(self.catalog),
            motion,
            navigation,
            MissionMap(),
            "OFF",
            {"robot": True, "cloud": False, "microphone": True, "tts": True},
            {"tof": True, "motion_opt_in": False, "cloud": False, "microphone": True, "tts": True},
        )

    def test_scenario_card_click_emits_selection(self):
        dashboard = ClassroomDashboard(1280, 840)
        canvas = dashboard.render(*self.make_render_args(dashboard))
        self.assertEqual(canvas.shape, (840, 1280, 3))
        scenario_region = next(
            region for region in dashboard.hit_regions if region["action"]["type"] == "scenario"
        )
        x1, y1, x2, y2 = scenario_region["rect"]
        dashboard.mouse_callback(cv2.EVENT_LBUTTONUP, (x1 + x2) // 2, (y1 + y2) // 2, 0)
        actions = dashboard.consume_actions()
        self.assertEqual(actions[0]["type"], "scenario")
        self.assertFalse(dashboard.show_scenario_menu)

    def test_full_camera_frame_is_aspect_fitted_without_cropping(self):
        dashboard = ClassroomDashboard(1280, 840)
        dashboard.show_scenario_menu = False
        dashboard.render(*self.make_render_args(dashboard, "target"))
        left, top, right, bottom = dashboard.last_render_metadata["video_rect"]
        displayed_ratio = (right - left) / (bottom - top)
        self.assertAlmostEqual(displayed_ratio, 1280 / 720, places=2)
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, dashboard.header_height)
        self.assertLessEqual(right, dashboard.width - dashboard.sidebar_width)
        self.assertLessEqual(bottom, dashboard.height - dashboard.caption_height)

    def test_clicking_detection_selects_target(self):
        dashboard = ClassroomDashboard(1280, 840)
        dashboard.show_scenario_menu = False
        dashboard.render(*self.make_render_args(dashboard, "target"))
        region = dashboard._detection_regions[0]
        x1, y1, x2, y2 = region["rect"]
        dashboard.mouse_callback(cv2.EVENT_LBUTTONUP, (x1 + x2) // 2, (y1 + y2) // 2, 0)
        action = dashboard.consume_actions()[0]
        self.assertEqual(action["type"], "select_target")
        self.assertEqual(action["value"], "person")

    def test_compact_scenario_menu_keeps_every_card_on_canvas(self):
        dashboard = ClassroomDashboard(960, 640)
        dashboard.render(*self.make_render_args(dashboard))
        cards = [
            region
            for region in dashboard.hit_regions
            if region["action"]["type"] == "scenario"
        ]
        self.assertEqual(len(cards), len(self.catalog))
        self.assertTrue(
            all(region["rect"][3] <= dashboard.height for region in cards)
        )

    def test_console_exposes_all_tts_controls(self):
        dashboard = ClassroomDashboard(1280, 840)
        dashboard.show_scenario_menu = False
        dashboard.render(*self.make_render_args(dashboard))
        action_types = [region["action"]["type"] for region in dashboard.hit_regions]
        self.assertIn("toggle_tts_mute", action_types)
        self.assertEqual(action_types.count("tts_rate"), 2)
        self.assertEqual(action_types.count("tts_volume"), 2)

    def test_patrol_mission_does_not_expose_irrelevant_target_clicks(self):
        dashboard = ClassroomDashboard(1280, 840)
        dashboard.show_scenario_menu = False
        dashboard.render(*self.make_render_args(dashboard, "exploration"))
        self.assertEqual(dashboard._detection_regions, [])
        action_types = [region["action"]["type"] for region in dashboard.hit_regions]
        self.assertNotIn("next_target", action_types)

    def test_student_role_hides_teacher_motion_controls(self):
        dashboard = ClassroomDashboard(1280, 840)
        dashboard.show_scenario_menu = False
        dashboard.render(*self.make_render_args(dashboard, "target"))
        role_region = next(
            region
            for region in dashboard.hit_regions
            if region["action"]["type"] == "toggle_role"
        )
        x1, y1, x2, y2 = role_region["rect"]
        dashboard.mouse_callback(
            cv2.EVENT_LBUTTONUP,
            (x1 + x2) // 2,
            (y1 + y2) // 2,
            0,
        )
        self.assertEqual(
            dashboard.consume_actions()[0],
            {"type": "toggle_role", "value": "student"},
        )
        dashboard.render(*self.make_render_args(dashboard, "target"))
        action_types = {region["action"]["type"] for region in dashboard.hit_regions}
        self.assertTrue(
            {"arm_toggle", "reset_estop", "add_waypoint", "return_home"}.isdisjoint(
                action_types
            )
        )


if __name__ == "__main__":
    unittest.main()
