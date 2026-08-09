from contextlib import ExitStack
import queue
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np

import robomaster_yolo_ai as app
from autonomy import DryRunBackend, MissionLogger, SafetyMotionController
from lego_vision import detect_aruco_lego_markers
from scenario_profiles import ScenarioCatalog


ROOT = Path(__file__).resolve().parents[1]


class RuntimeFailureTests(unittest.TestCase):
    def test_conversation_history_is_reused_and_isolated_by_scenario(self):
        class FakeCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                turn = len(self.calls)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=f"reply {turn}")
                        )
                    ]
                )

        completions = FakeCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        request = {
            "scenario_id": "exploration",
            "labels": ["chair"],
            "target_label": "person",
        }
        with app.state_lock:
            app.conversation_histories["exploration"] = []
            app.conversation_histories["rescue"] = []
        first = app.generate_conversation_reply(client, request, "What is here?")
        second = app.generate_conversation_reply(client, request, "Anything else?")

        self.assertEqual((first, second), ("reply 1", "reply 2"))
        second_messages = completions.calls[1]["messages"]
        self.assertIn(
            {"role": "user", "content": "What is here?"},
            second_messages,
        )
        self.assertIn(
            {"role": "assistant", "content": "reply 1"},
            second_messages,
        )
        self.assertEqual(app.conversation_histories["rescue"], [])
        with app.state_lock:
            app.conversation_histories["exploration"] = []

    def test_openai_factory_failure_returns_local_only_mode(self):
        def unavailable():
            raise RuntimeError("missing credential")

        client, error = app.create_openai_client(unavailable)
        self.assertIsNone(client)
        self.assertIn("missing credential", str(error))

    def test_duplicate_tof_mapping_is_rejected_before_backend_start(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            app.validate_tof_layout("front,front,right,rear", 4)

    def test_valid_tof_mapping_is_normalized(self):
        self.assertEqual(
            app.validate_tof_layout(" Front, Left, Right, Rear ", 4),
            ("front", "left", "right", "rear"),
        )

    def test_valid_lego_target_survives_missed_frame(self):
        selected = app.choose_initial_lego_target(
            "lego_marker_7",
            ["lego_blue"],
        )
        self.assertEqual(selected, "lego_marker_7")

    def test_lego_target_can_be_unselected_until_a_real_lego_is_seen(self):
        self.assertFalse(app.is_lego_target_label(None))
        self.assertEqual(
            app.choose_initial_lego_target(None, ["lego_marker_3"]),
            "lego_marker_3",
        )

    def test_profile_target_filter_combines_allowlist_and_lego_type(self):
        catalog = ScenarioCatalog.load(ROOT / "scenarios.json")
        self.assertTrue(app.profile_accepts_target(catalog.get("follow"), "person"))
        self.assertFalse(app.profile_accepts_target(catalog.get("follow"), "tv"))
        self.assertTrue(
            app.profile_accepts_target(catalog.get("lego"), "lego_marker_7")
        )
        self.assertFalse(app.profile_accepts_target(catalog.get("lego"), "person"))

    def test_rescue_person_disarms_even_when_narration_is_irrelevant(self):
        catalog = ScenarioCatalog.load(ROOT / "scenarios.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = SafetyMotionController(
                DryRunBackend(),
                motion_requested=True,
                logger=MissionLogger(Path(temp_dir) / "events.jsonl"),
            )
            controller.connect()
            try:
                self.assertTrue(controller.arm())
                seen = app.enforce_rescue_person_stop(
                    catalog.get("rescue"),
                    ["person"],
                    controller,
                )
                self.assertTrue(seen)
                self.assertFalse(controller.armed)
            finally:
                controller.close()

    def test_closed_preview_is_reported_invisible(self):
        with mock.patch.object(app.cv2, "getWindowProperty", return_value=0):
            self.assertFalse(app.preview_window_is_visible("preview"))
        with mock.patch.object(
            app.cv2,
            "getWindowProperty",
            side_effect=cv2.error("closed"),
        ):
            self.assertFalse(app.preview_window_is_visible("preview"))

    def test_tts_initialization_failure_disables_future_queueing(self):
        app.tts_failed.clear()
        app.tts_available.clear()
        while True:
            try:
                app.speech_queue.get_nowait()
            except queue.Empty:
                break
        try:
            with mock.patch.object(app.pyttsx3, "init", side_effect=RuntimeError("no voice")):
                app.speech_worker()
            self.assertTrue(app.tts_failed.is_set())
            self.assertFalse(app.speak("This must not be queued"))
            self.assertTrue(app.speech_queue.empty())
        finally:
            app.tts_failed.clear()

    def test_tts_runtime_failure_drains_queue_and_disables_worker(self):
        class FailingEngine:
            def setProperty(self, *_args):
                pass

            def say(self, _text):
                pass

            def runAndWait(self):
                raise RuntimeError("speaker disconnected")

            def stop(self):
                pass

        app.shutdown_event.clear()
        app.tts_failed.clear()
        app.tts_available.clear()
        while True:
            try:
                app.speech_queue.get_nowait()
            except queue.Empty:
                break
        app.speech_queue.put_nowait("first")
        app.speech_queue.put_nowait("second")
        try:
            with mock.patch.object(app.pyttsx3, "init", return_value=FailingEngine()):
                app.speech_worker()
            self.assertTrue(app.tts_failed.is_set())
            self.assertFalse(app.tts_available.is_set())
            self.assertTrue(app.speech_queue.empty())
        finally:
            app.tts_failed.clear()
            app.tts_available.clear()

    def test_main_starts_local_only_without_openai_credentials(self):
        class FakeMSS:
            monitors = [
                {},
                {"left": 0, "top": 0, "width": 1280, "height": 900},
            ]

            def grab(self, _monitor):
                return np.zeros((720, 1280, 4), dtype=np.uint8)

            def close(self):
                pass

        class FakeYolo:
            names = {}

            def __init__(self, _model):
                pass

            def __call__(self, *_args, **_kwargs):
                return []

        with tempfile.TemporaryDirectory() as temp_dir:
            app.shutdown_event.clear()
            app.cloud_available.clear()
            patchers = (
                mock.patch.object(app, "create_openai_client", return_value=(None, RuntimeError("no key"))),
                mock.patch.object(app, "YOLO", FakeYolo),
                mock.patch.object(app, "create_screen_capture", return_value=FakeMSS()),
                mock.patch.object(app, "speech_worker", return_value=None),
                mock.patch.object(app, "preview_window_is_visible", return_value=False),
                mock.patch.object(app, "exclude_preview_from_capture"),
                mock.patch.object(app.cv2, "namedWindow"),
                mock.patch.object(app.cv2, "resizeWindow"),
                mock.patch.object(app.cv2, "moveWindow"),
                mock.patch.object(app.cv2, "setMouseCallback"),
                mock.patch.object(app.cv2, "setWindowProperty"),
                mock.patch.object(app.cv2, "imshow"),
                mock.patch.object(app.cv2, "waitKey", return_value=-1),
                mock.patch.object(app.cv2, "destroyAllWindows"),
                mock.patch.object(app.sd, "query_devices", side_effect=RuntimeError("no mic")),
                mock.patch.object(app.sd, "stop"),
            )
            with ExitStack() as stack:
                for patcher in patchers:
                    stack.enter_context(patcher)
                app.main(
                    [
                        "--disable-lego-model",
                        "--mission-log",
                        str(Path(temp_dir) / "events.jsonl"),
                        "--mission-map",
                        str(Path(temp_dir) / "map.json"),
                        "--scenarios",
                        str(ROOT / "scenarios.json"),
                    ]
                )
        self.assertFalse(app.cloud_available.is_set())
        app.shutdown_event.clear()

    def test_primary_detector_load_failure_exits_after_safe_backend_close(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                app,
                "YOLO",
                side_effect=RuntimeError("bad checkpoint"),
            ):
                with self.assertRaisesRegex(SystemExit, "could not be loaded"):
                    app.main(
                        [
                            "--disable-lego-model",
                            "--mission-log",
                            str(Path(temp_dir) / "events.jsonl"),
                            "--mission-map",
                            str(Path(temp_dir) / "map.json"),
                            "--scenarios",
                            str(ROOT / "scenarios.json"),
                        ]
                    )


class LegacyArucoTests(unittest.TestCase):
    def test_legacy_parameter_factory_is_used(self):
        class LegacyAruco:
            DICT_4X4_50 = 0

            def __init__(self):
                self.created = False

            def getPredefinedDictionary(self, _dictionary_id):
                return object()

            def DetectorParameters_create(self):
                self.created = True
                return object()

            def detectMarkers(self, _frame, _dictionary, parameters=None):
                self.parameters = parameters
                return [], None, []

        legacy = LegacyAruco()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with mock.patch.object(cv2, "aruco", legacy):
            self.assertEqual(detect_aruco_lego_markers(frame), [])
        self.assertTrue(legacy.created)

    def test_broken_modern_parameter_constructor_falls_back_to_factory(self):
        class MixedAruco:
            DICT_4X4_50 = 0

            def __init__(self):
                self.created = False

            def getPredefinedDictionary(self, _dictionary_id):
                return object()

            def DetectorParameters(self):
                raise TypeError("constructor unavailable")

            def DetectorParameters_create(self):
                self.created = True
                return object()

            def detectMarkers(self, _frame, _dictionary, parameters=None):
                return [], None, []

        aruco = MixedAruco()
        with mock.patch.object(cv2, "aruco", aruco):
            self.assertEqual(
                detect_aruco_lego_markers(np.zeros((80, 80, 3), dtype=np.uint8)),
                [],
            )
        self.assertTrue(aruco.created)


if __name__ == "__main__":
    unittest.main()
