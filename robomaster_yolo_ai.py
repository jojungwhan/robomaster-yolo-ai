import argparse
import atexit
import base64
import ctypes
import queue
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import cv2
import mss
import numpy as np
import pyttsx3
import sounddevice as sd
from openai import OpenAI
from ultralytics import YOLO

from autonomy import (
    AutonomousPlanner,
    DryRunBackend,
    MissionLogger,
    RoboMasterBackend,
    SafetyMotionController,
    VALID_TOF_DIRECTIONS,
)
from dashboard import ClassroomDashboard
from lego_identification import (
    AsyncLegoIdentificationDetector,
    LegoIdentificationDetector,
)
from lego_vision import (
    HELP_SIGNAL_LABEL,
    ConsecutiveDetectionGate,
    detect_lego,
    detect_red_cross_signal,
)
from navigation import GimbalTracker, MissionMap, MissionNavigator, NavigationState
from scenario_profiles import ScenarioCatalog


CHAT_MODEL = "gpt-4o-mini"
TRANSCRIPTION_MODELS = ("gpt-4o-mini-transcribe", "whisper-1")
RECORD_SECONDS = 5
MAX_HISTORY_MESSAGES = 12
DEFAULT_LEGO_MODEL = (
    Path(__file__).resolve().parent
    / "models"
    / "lego-identification"
    / "FinalCoShSi.pt"
)
DEFAULT_SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios.json"
DEFAULT_MISSION_MAP_PATH = Path(__file__).resolve().parent / "mission_map.json"
scenario_catalog = ScenarioCatalog.load(DEFAULT_SCENARIOS_PATH)
SCENARIOS = scenario_catalog.as_legacy_dict()
SCENARIO_KEYS = scenario_catalog.key_map

shutdown_event = threading.Event()
voice_busy = threading.Event()
tts_speaking = threading.Event()
tts_available = threading.Event()
tts_failed = threading.Event()
cloud_available = threading.Event()
speech_queue = queue.Queue(maxsize=32)
voice_requests = queue.Queue(maxsize=1)
scene_requests = queue.Queue(maxsize=1)
state_lock = threading.Lock()

runtime_state = {
    "scenario_id": scenario_catalog.default.id,
    "target_label": "person",
    "auto_narration": True,
    "voice_status": "Press V, Space, or F8 to talk",
    "student_caption": "",
    "robot_caption": "",
    "cloud_status": "checking",
    "tts_status": "starting",
    "tts_muted": False,
    "tts_rate": 150,
    "tts_volume": 1.0,
    "mission_paused": False,
    "ui_role": "teacher",
    "navigation_state": NavigationState.DISARMED.value,
    "navigation_reason": "motion disarmed",
    "gimbal_status": "disabled",
    "yolo_inference_ms": 0.0,
}

conversation_histories = {scenario_id: [] for scenario_id in SCENARIOS}


def get_scenario_profile(scenario_id):
    return scenario_catalog.get(scenario_id)


def scenario_tracks_visual_target(scenario_id):
    return get_scenario_profile(scenario_id).navigation_policy in ("target", "lego")


def set_voice_status(status):
    with state_lock:
        runtime_state["voice_status"] = status


def get_runtime_state():
    with state_lock:
        return dict(runtime_state)


def play_beep(frequency=900, duration_ms=120):
    if sys.platform != "win32":
        return

    try:
        import winsound

        winsound.Beep(frequency, duration_ms)
    except Exception:
        pass


def global_voice_hotkey_pressed():
    """Detect V or F8 even when the OpenCV window does not have focus."""
    if sys.platform != "win32":
        return False

    user32 = ctypes.windll.user32
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    virtual_keys = (ord("V"), 0x77)  # V and F8
    return any(user32.GetAsyncKeyState(key) & 0x0001 for key in virtual_keys)


def global_emergency_stop_pressed():
    """Escape is a global emergency stop while the application is running."""
    if sys.platform != "win32":
        return False

    user32 = ctypes.windll.user32
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    return bool(user32.GetAsyncKeyState(0x1B) & 0x0001)


def request_voice_turn(state, labels):
    if not cloud_available.is_set():
        set_voice_status("Voice conversation unavailable - OpenAI key not configured")
        speak("Cloud conversation is unavailable. Local vision is still running.")
        return False
    if voice_busy.is_set():
        set_voice_status("Already processing a voice turn")
        return False

    voice_busy.set()
    set_voice_status("Talk command accepted - preparing microphone...")
    print("[Voice]: Talk command accepted. Waiting for speech output to finish.")
    play_beep(700, 80)

    try:
        voice_requests.put_nowait(
            {
                "scenario_id": state["scenario_id"],
                "labels": labels,
                "target_label": state["target_label"],
            }
        )
        return True
    except queue.Full:
        voice_busy.clear()
        set_voice_status("Voice queue busy - try again")
        return False


def format_spoken_list(labels):
    labels = list(labels)
    if not labels:
        return "nothing recognizable"
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def speak(text):
    text = (text or "").strip()
    if text:
        print(f"[Robot Voice]: {text}")
        with state_lock:
            runtime_state["robot_caption"] = text
            runtime_state["robot_caption_at"] = time.time()
            muted = runtime_state["tts_muted"]
        if muted:
            return False
        if tts_failed.is_set():
            return False
        try:
            speech_queue.put_nowait(text)
            return True
        except queue.Full:
            with state_lock:
                runtime_state["tts_status"] = "busy - speech queue full"
            return False
    return False


def disable_tts(error):
    tts_available.clear()
    tts_failed.set()
    with state_lock:
        runtime_state["tts_status"] = f"unavailable: {error}"
    while True:
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break


def speech_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
        tts_available.set()
        with state_lock:
            runtime_state["tts_status"] = "ready"
        print("[TTS]: Speech engine ready.")
    except Exception as error:
        disable_tts(error)
        print(f"Text-to-speech initialization error: {error}")
        return

    while not shutdown_event.is_set():
        try:
            text = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if text is None:
            break

        try:
            tts_speaking.set()
            current_state = get_runtime_state()
            if current_state["tts_muted"]:
                continue
            engine.setProperty("rate", current_state["tts_rate"])
            engine.setProperty("volume", current_state["tts_volume"])
            engine.say(text)
            engine.runAndWait()
        except Exception as error:
            print(f"Text-to-speech error: {error}")
            disable_tts(error)
            break
        finally:
            tts_speaking.clear()

    try:
        engine.stop()
    except Exception:
        pass


def record_microphone_to_wav(duration_seconds):
    input_device = sd.query_devices(kind="input")
    sample_rate = int(input_device["default_samplerate"])
    sample_count = int(duration_seconds * sample_rate)

    audio = sd.rec(
        sample_count,
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = Path(temp_file.name)
    temp_file.close()

    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

    return audio_path


def transcribe_audio(openai_client, audio_path):
    errors = []

    for model in TRANSCRIPTION_MODELS:
        try:
            with audio_path.open("rb") as audio_file:
                transcription = openai_client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                )

            if isinstance(transcription, str):
                return transcription.strip()
            return (transcription.text or "").strip()
        except Exception as error:
            errors.append(f"{model}: {error}")

    raise RuntimeError("; ".join(errors))


def build_conversation_system_prompt(scenario_id, labels, target_label):
    prompt = SCENARIOS[scenario_id]["system_prompt"]
    label_text = ", ".join(labels) if labels else "none"
    prompt += f" Current unverified computer-vision labels: {label_text}."

    if scenario_tracks_visual_target(scenario_id):
        prompt += f" The selected visual target is {target_label or 'not selected'}."

    return prompt


def generate_conversation_reply(openai_client, request, user_text):
    """Run one chat turn and persist bounded history for the selected mission."""
    scenario_id = request["scenario_id"]
    system_prompt = build_conversation_system_prompt(
        scenario_id,
        request["labels"],
        request["target_label"],
    )
    with state_lock:
        recent_history = list(
            conversation_histories[scenario_id][-MAX_HISTORY_MESSAGES:]
        )

    messages = [
        {"role": "system", "content": system_prompt},
        *recent_history,
        {"role": "user", "content": user_text},
    ]
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=160,
    )
    reply = (response.choices[0].message.content or "").strip()

    with state_lock:
        conversation_histories[scenario_id].extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ]
        )
        conversation_histories[scenario_id] = conversation_histories[scenario_id][
            -MAX_HISTORY_MESSAGES:
        ]
    return reply


def voice_conversation_worker(openai_client):
    while not shutdown_event.is_set():
        try:
            request = voice_requests.get(timeout=0.2)
        except queue.Empty:
            continue

        if request is None:
            break

        audio_path = None
        try:
            speech_wait_deadline = time.monotonic() + 15
            while (
                not tts_failed.is_set()
                and (tts_speaking.is_set() or not speech_queue.empty())
                and not shutdown_event.is_set()
                and time.monotonic() < speech_wait_deadline
            ):
                time.sleep(0.05)

            if tts_speaking.is_set() or not speech_queue.empty():
                print("[Voice]: Speech queue wait timed out; starting microphone anyway.")

            set_voice_status(f"Recording for {RECORD_SECONDS} seconds...")
            print(f"[Voice]: Recording for {RECORD_SECONDS} seconds. Speak now.")

            play_beep(1000, 150)

            audio_path = record_microphone_to_wav(RECORD_SECONDS)
            set_voice_status("Transcribing...")
            user_text = transcribe_audio(openai_client, audio_path)

            if not user_text:
                set_voice_status("No speech detected - press V to retry")
                speak("I did not hear anything. Press V and try again.")
                continue

            print(f"[You]: {user_text}")
            with state_lock:
                runtime_state["student_caption"] = user_text
                runtime_state["student_caption_at"] = time.time()
            set_voice_status("Thinking...")
            reply = generate_conversation_reply(openai_client, request, user_text)

            set_voice_status("Press V for another turn")
            speak(reply)
        except Exception as error:
            print(f"Voice conversation error: {error}")
            set_voice_status("Voice error - press V to retry")
            speak("I could not process that voice turn.")
        finally:
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)
            voice_busy.clear()


def scene_description_worker(openai_client):
    while not shutdown_event.is_set():
        try:
            request = scene_requests.get(timeout=0.2)
        except queue.Empty:
            continue

        if request is None:
            break

        scenario_id = request["scenario_id"]
        labels = request["labels"]
        target_label = request["target_label"]

        try:
            encode_ok, buffer = cv2.imencode(
                ".jpg",
                request["frame"],
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            if not encode_ok:
                raise RuntimeError("OpenCV could not encode the scene image.")

            base64_image = base64.b64encode(buffer).decode("utf-8")
            label_text = ", ".join(labels)
            mission_prompt = SCENARIOS[scenario_id]["scene_prompt"]
            target_context = (
                f" Selected target: {target_label}."
                if scenario_tracks_visual_target(scenario_id)
                else ""
            )

            response = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"You are operating in {SCENARIOS[scenario_id]['name']} "
                                    f"mode. Computer vision reported these unverified labels: "
                                    f"{label_text}.{target_context} {mission_prompt}"
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=80,
            )
            description = (response.choices[0].message.content or "").strip()
            print(f"[Scene AI]: {description}")

            if not voice_busy.is_set():
                speak(description)
        except Exception as error:
            print(f"OpenAI scene description error: {error}")


def exclude_preview_from_capture(preview_window):
    if sys.platform != "win32":
        return

    cv2.waitKey(1)
    user32 = ctypes.windll.user32
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    user32.SetWindowDisplayAffinity.restype = ctypes.c_bool

    preview_handle = user32.FindWindowW(None, preview_window)
    if preview_handle:
        wda_exclude_from_capture = 0x11
        wda_monitor_only = 0x01
        if not user32.SetWindowDisplayAffinity(
            preview_handle, wda_exclude_from_capture
        ):
            user32.SetWindowDisplayAffinity(preview_handle, wda_monitor_only)


def preview_window_is_visible(preview_window):
    try:
        return cv2.getWindowProperty(preview_window, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def create_screen_capture():
    """Create an MSS capture object across both the v9 and v10 public APIs."""
    capture_type = getattr(mss, "MSS", None)
    if capture_type is not None:
        return capture_type()
    return mss.mss()


def calculate_target_guidance(detections, target_label, frame_shape):
    if not target_label:
        return "PRESS T TO SELECT A TARGET", None

    matches = [item for item in detections if item["label"] == target_label]
    if not matches:
        return f"SEARCH FOR {target_label.upper()}", None

    target = max(matches, key=lambda item: item["confidence"])
    x1, y1, x2, y2 = target["box"]
    frame_height, frame_width = frame_shape[:2]
    target_center_x = (x1 + x2) / 2
    target_height_ratio = (y2 - y1) / frame_height

    if target_center_x < frame_width * 0.42:
        guidance = "TURN LEFT"
    elif target_center_x > frame_width * 0.58:
        guidance = "TURN RIGHT"
    elif target_height_ratio < 0.35:
        guidance = "MOVE FORWARD"
    else:
        guidance = "STOP - TARGET CLOSE"

    return guidance, target


def draw_detection(frame, detection, selected, annotation):
    x1, y1, x2, y2 = detection["box"]
    label = detection["label"]
    confidence = detection["confidence"]
    if detection["label"] == HELP_SIGNAL_LABEL:
        box_color = (0, 0, 255)
    else:
        box_color = (0, 165, 255) if selected else (0, 255, 0)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        box_color,
        annotation["box_thickness"],
        cv2.LINE_AA,
    )

    label_text = f"{label} {confidence:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        annotation["font_scale"],
        annotation["font_thickness"],
    )
    padding = annotation["label_padding"]
    text_x = max(0, min(x1, frame.shape[1] - text_width - 2 * padding))
    label_top = max(0, y1 - text_height - 2 * padding)
    label_bottom = label_top + text_height + 2 * padding

    cv2.rectangle(
        frame,
        (text_x, label_top),
        (text_x + text_width + 2 * padding, label_bottom),
        box_color,
        -1,
    )
    cv2.putText(
        frame,
        label_text,
        (text_x + padding, label_top + text_height + padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        annotation["font_scale"],
        (0, 0, 0),
        annotation["font_thickness"],
        cv2.LINE_AA,
    )


def cycle_target(current_target, labels):
    labels = sorted(set(labels))
    if not labels:
        return current_target

    if current_target not in labels:
        return labels[0]

    current_index = labels.index(current_target)
    return labels[(current_index + 1) % len(labels)]


def is_lego_target_label(label):
    return (
        isinstance(label, str)
        and label.startswith("lego_")
        and not label.startswith("lego_signal_")
    )


def choose_initial_lego_target(current_target, labels):
    """Choose once on mode entry; never replace a valid target after a missed frame."""
    if is_lego_target_label(current_target):
        return current_target
    lego_labels = sorted(label for label in labels if is_lego_target_label(label))
    marker_labels = [label for label in lego_labels if label.startswith("lego_marker_")]
    if marker_labels:
        return marker_labels[0]
    return lego_labels[0] if lego_labels else current_target


def profile_accepts_target(profile, label):
    if not profile.accepts_target(label):
        return False
    if profile.navigation_policy == "lego":
        return is_lego_target_label(label)
    return True


def validate_tof_layout(layout_text, min_tof_count):
    layout = tuple(
        item.strip().lower() for item in layout_text.split(",") if item.strip()
    )
    if "front" not in layout:
        raise ValueError("--tof-layout must include a front sensor.")
    if len(set(layout)) != len(layout):
        raise ValueError("--tof-layout directions must be unique.")
    unsupported = set(layout) - VALID_TOF_DIRECTIONS
    if unsupported:
        raise ValueError(
            "--tof-layout contains unsupported directions: "
            + ", ".join(sorted(unsupported))
        )
    if min_tof_count < 1 or min_tof_count > len(layout):
        raise ValueError("--min-tof-count must fit the configured --tof-layout.")
    return layout


def create_openai_client(client_factory=OpenAI):
    try:
        return client_factory(), None
    except Exception as error:
        return None, error


def enforce_rescue_person_stop(profile, labels, motion_controller):
    person_seen = profile.navigation_policy == "rescue" and "person" in labels
    if person_seen and motion_controller.armed:
        motion_controller.disarm(
            "person detected in rescue mode; re-arm after assessment"
        )
    return person_seen


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="RoboMaster vision, voice, LEGO, and safety-gated autonomy demo"
    )
    parser.add_argument(
        "--motion-backend",
        choices=("dry-run", "robomaster"),
        default="dry-run",
        help="Use dry-run unless a compatible SDK-controlled robot and ToF sensors are ready.",
    )
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="Permit arming with M after all ToF safety checks pass.",
    )
    parser.add_argument(
        "--enable-gimbal-tracking",
        action="store_true",
        help="Track the locked target with the gimbal only while motion is armed.",
    )
    parser.add_argument(
        "--conn-type",
        choices=("ap", "sta", "rndis"),
        default="ap",
        help="RoboMaster SDK connection type for the hardware backend.",
    )
    parser.add_argument(
        "--tof-layout",
        default="front,left,right,rear",
        help="Comma-separated physical direction of each ToF channel.",
    )
    parser.add_argument(
        "--min-tof-count",
        type=int,
        default=4,
        help="Minimum live ToF channels required before motion can arm.",
    )
    parser.add_argument(
        "--wall-clearance-mm",
        type=float,
        default=700,
    )
    parser.add_argument(
        "--person-clearance-mm",
        type=float,
        default=1200,
    )
    parser.add_argument(
        "--mission-log",
        default="mission_events.jsonl",
    )
    parser.add_argument(
        "--scenarios",
        default=str(DEFAULT_SCENARIOS_PATH),
        help="Path to the version-1 scenario profile JSON file.",
    )
    parser.add_argument(
        "--mission-map",
        default=str(DEFAULT_MISSION_MAP_PATH),
        help="Path used to load and save odometry, obstacles, and waypoints.",
    )
    parser.add_argument(
        "--resume-map",
        action="store_true",
        help="Resume a saved map; otherwise the current position becomes a new home.",
    )
    parser.add_argument("--dashboard-width", type=int, default=1280)
    parser.add_argument("--dashboard-height", type=int, default=840)
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="Ultralytics detection checkpoint; use a custom model for domain classes.",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.20,
        help="Minimum primary object-detection confidence.",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="Primary YOLO inference size; smaller values are faster but less detailed.",
    )
    parser.add_argument(
        "--yolo-interval",
        type=float,
        default=0.10,
        help="Minimum seconds after an inference before running YOLO again.",
    )
    parser.add_argument(
        "--yolo-device",
        help="Optional Ultralytics device such as 0, cpu, or mps.",
    )
    parser.add_argument(
        "--lego-model",
        default=str(DEFAULT_LEGO_MODEL),
        help="Path to Lego-Identification FinalCoShSi.pt.",
    )
    parser.add_argument(
        "--disable-lego-model",
        action="store_true",
        help="Use only the OpenCV LEGO and semantic-pattern detectors.",
    )
    parser.add_argument(
        "--lego-model-confidence",
        type=float,
        default=0.30,
        help="Minimum Lego-Identification confidence.",
    )
    parser.add_argument(
        "--lego-model-interval",
        type=float,
        default=0.75,
        help="Seconds between LEGO model inferences; cached boxes are shown between runs.",
    )
    parser.add_argument(
        "--lego-model-imgsz",
        type=int,
        default=512,
        help="Lego-Identification inference image size.",
    )
    parser.add_argument("--screen", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    global scenario_catalog, SCENARIOS, SCENARIO_KEYS, conversation_histories
    args = parse_args(argv)
    try:
        scenario_catalog = ScenarioCatalog.load(args.scenarios)
    except Exception as error:
        raise SystemExit(f"Could not load scenario profiles: {error}") from error
    SCENARIOS = scenario_catalog.as_legacy_dict()
    SCENARIO_KEYS = scenario_catalog.key_map
    conversation_histories = {
        profile.id: list(conversation_histories.get(profile.id, ()))
        for profile in scenario_catalog
    }
    with state_lock:
        runtime_state["scenario_id"] = scenario_catalog.default.id
        preferred = scenario_catalog.default.preferred_target
        if preferred:
            runtime_state["target_label"] = preferred
    if args.dashboard_width < 960 or args.dashboard_height < 640:
        raise SystemExit("Dashboard size must be at least 960x640.")
    if not 0 <= args.yolo_confidence <= 1:
        raise SystemExit("--yolo-confidence must be between 0 and 1.")
    if args.yolo_imgsz < 160:
        raise SystemExit("--yolo-imgsz must be at least 160.")
    if args.yolo_interval < 0:
        raise SystemExit("--yolo-interval cannot be negative.")

    try:
        tof_layout = validate_tof_layout(args.tof_layout, args.min_tof_count)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.motion_backend == "robomaster":
        motion_backend = RoboMasterBackend(conn_type=args.conn_type)
    else:
        motion_backend = DryRunBackend()

    motion_controller = SafetyMotionController(
        motion_backend,
        motion_requested=args.enable_motion,
        tof_layout=tof_layout,
        min_tof_count=args.min_tof_count,
        wall_clearance_mm=args.wall_clearance_mm,
        person_clearance_mm=args.person_clearance_mm,
        logger=MissionLogger(args.mission_log),
    )
    try:
        motion_controller.connect()
    except Exception as error:
        motion_controller.close()
        raise SystemExit(f"Motion backend could not start safely: {error}") from error
    atexit.register(motion_controller.close)

    planner = AutonomousPlanner()
    if args.resume_map:
        try:
            mission_map = MissionMap.load(args.mission_map)
        except Exception as error:
            print(f"[Map]: Could not load {args.mission_map}; starting a new map: {error}")
            mission_map = MissionMap()
    else:
        mission_map = MissionMap()
    navigator = MissionNavigator(planner=planner, mission_map=mission_map)
    gimbal_tracker = GimbalTracker()
    try:
        yolo_model = YOLO(args.yolo_model)
    except Exception as error:
        motion_controller.close()
        try:
            atexit.unregister(motion_controller.close)
        except Exception:
            pass
        raise SystemExit(
            f"Primary YOLO model could not be loaded ({args.yolo_model}): {error}"
        ) from error
    lego_detector = None
    if args.disable_lego_model:
        lego_model_status = "OFF (--disable-lego-model)"
    else:
        try:
            lego_detector = AsyncLegoIdentificationDetector(
                LegoIdentificationDetector(
                    args.lego_model,
                    confidence=args.lego_model_confidence,
                    inference_size=args.lego_model_imgsz,
                    interval_seconds=args.lego_model_interval,
                )
            )
            lego_model_status = lego_detector.summary
            print(
                f"[LEGO Model]: Loaded {args.lego_model} - {lego_model_status}."
            )
        except FileNotFoundError:
            lego_model_status = "OFF (run setup_lego_identification.ps1)"
            print(
                f"[LEGO Model]: {args.lego_model} was not found. "
                "Run .\\setup_lego_identification.ps1 to install the pinned model."
            )
        except Exception as error:
            lego_model_status = f"OFF ({type(error).__name__})"
            print(f"[LEGO Model]: Disabled safely: {error}")
    openai_client, openai_error = create_openai_client()
    if openai_client is not None:
        cloud_available.set()
        with state_lock:
            runtime_state["cloud_status"] = "ready"
    else:
        cloud_available.clear()
        with state_lock:
            runtime_state["cloud_status"] = "unavailable"
            runtime_state["voice_status"] = "Local vision only - OpenAI key not configured"
        print(
            "[OpenAI]: Cloud features disabled; local vision remains active: "
            f"{openai_error}"
        )

    sct = create_screen_capture()
    max_screen = len(sct.monitors) - 1
    if args.screen < 1 or args.screen > max_screen:
        sct.close()
        motion_controller.close()
        try:
            atexit.unregister(motion_controller.close)
        except Exception:
            pass
        raise SystemExit(f"--screen must be between 1 and {max_screen}.")
    monitor = dict(sct.monitors[args.screen])

    preview_window = "RoboMaster S1 - YOLO CV Pipeline"
    preview_width = min(args.dashboard_width, monitor["width"])
    preview_height = min(args.dashboard_height, monitor["height"] - 40)
    if preview_width < 960 or preview_height < 600:
        motion_controller.close()
        sct.close()
        try:
            atexit.unregister(motion_controller.close)
        except Exception:
            pass
        raise SystemExit(
            "Selected screen is too small for the mission dashboard; "
            "choose a screen with at least 960x640 usable pixels."
        )
    dashboard = ClassroomDashboard(preview_width, preview_height)
    preview_x = monitor["left"] + max(0, monitor["width"] - preview_width - 20)
    preview_y = monitor["top"] + 20

    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview_window, preview_width, preview_height)
    cv2.moveWindow(preview_window, preview_x, preview_y)
    cv2.setMouseCallback(preview_window, dashboard.mouse_callback)

    try:
        cv2.setWindowProperty(preview_window, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        print("OpenCV could not enable always-on-top mode on this system.")

    exclude_preview_from_capture(preview_window)

    annotation_scale = max(
        monitor["width"] / max(args.dashboard_width - dashboard.sidebar_width, 1),
        monitor["height"] / max(args.dashboard_height - dashboard.caption_height, 1),
    )
    annotation = {
        "box_thickness": max(2, round(2 * annotation_scale)),
        "font_scale": 0.55 * annotation_scale,
        "font_thickness": max(1, round(annotation_scale)),
        "label_padding": max(2, round(4 * annotation_scale)),
    }

    workers = [threading.Thread(target=speech_worker, name="speech", daemon=True)]
    if openai_client is not None:
        workers.extend(
            [
                threading.Thread(
                    target=voice_conversation_worker,
                    args=(openai_client,),
                    name="voice-conversation",
                    daemon=True,
                ),
                threading.Thread(
                    target=scene_description_worker,
                    args=(openai_client,),
                    name="scene-description",
                    daemon=True,
                ),
            ]
        )
    for worker in workers:
        worker.start()

    try:
        sd.query_devices(kind="input")
        microphone_ready = True
    except Exception as error:
        microphone_ready = False
        print(f"[Microphone]: No usable input device detected: {error}")

    print(
        "YOLO Mission System Active. "
        + ("OpenAI cloud enabled." if openai_client else "Local-only mode.")
    )
    print(
        "Keys: 0-9 Scenario | S Scenario menu | V/Space/F8 Talk | L Read objects | "
        "T Next target | W Waypoint | H Home | P Pause | M Arm | E/Esc Stop | Q Quit"
    )
    print(
        f"[Motion]: backend={args.motion_backend}, requested={args.enable_motion}, "
        f"ToF layout={','.join(tof_layout)}. Motion starts DISARMED."
    )
    speak(
        "Mission system online. Select a scenario card to begin. "
        "Motion remains disarmed until a teacher completes preflight and presses arm."
    )

    last_ai_check = time.time()
    ai_interval = 12
    yolo_detections = []
    yolo_inference_sequence = 0
    last_yolo_inference_at = float("-inf")
    last_guidance = None
    last_guidance_spoken = 0
    last_rescue_signature = None
    last_rescue_announcement = 0
    previous_scenario = scenario_catalog.default.id
    last_position_log = 0
    help_signal_gate = ConsecutiveDetectionGate(required_frames=3, release_frames=8)
    pending_help_check = False
    previous_rescue_person_seen = False
    previous_navigation_state = None
    last_map_save = 0
    preview_rendered = False
    gimbal_active = False
    dashboard.add_event("Mission console started")

    try:
        while True:
            if preview_rendered and not preview_window_is_visible(preview_window):
                motion_controller.emergency_stop("preview window closed")
                print("[Safety]: Preview window closed; motion stopped.")
                break
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            state = get_runtime_state()
            profile = get_scenario_profile(state["scenario_id"])
            if state["scenario_id"] != previous_scenario:
                previous_scenario = state["scenario_id"]
                last_rescue_signature = None
                last_rescue_announcement = 0
                help_signal_gate.reset()
                pending_help_check = False

            capture_time = time.monotonic()
            if capture_time - last_yolo_inference_at >= args.yolo_interval:
                inference_started_at = time.monotonic()
                yolo_options = {
                    "stream": True,
                    "verbose": False,
                    "conf": args.yolo_confidence,
                    "imgsz": args.yolo_imgsz,
                }
                if args.yolo_device:
                    yolo_options["device"] = args.yolo_device
                results = yolo_model(frame, **yolo_options)
                fresh_detections = []
                yolo_inference_sequence += 1
                observation_id = f"yolo:{yolo_inference_sequence}"
                for result in results:
                    names = getattr(result, "names", yolo_model.names)
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        fresh_detections.append(
                            {
                                "box": (x1, y1, x2, y2),
                                "label": names[class_id],
                                "confidence": float(box.conf[0]),
                                "source": "yolo",
                                "observation_id": observation_id,
                            }
                        )
                yolo_detections = fresh_detections
                last_yolo_inference_at = time.monotonic()
                with state_lock:
                    runtime_state["yolo_inference_ms"] = (
                        last_yolo_inference_at - inference_started_at
                    ) * 1000
            detections = [dict(item) for item in yolo_detections]

            if profile.use_lego_vision:
                if lego_detector is not None:
                    try:
                        detections.extend(lego_detector.detect(frame))
                    except Exception as error:
                        lego_model_status = f"OFF ({type(error).__name__})"
                        print(f"[LEGO Model]: Inference failed; disabled: {error}")
                        lego_detector.close()
                        lego_detector = None

                if profile.navigation_policy == "rescue":
                    detections.extend(detect_red_cross_signal(frame))
                else:
                    detections.extend(detect_lego(frame))

            if profile.navigation_policy == "lego":

                lego_labels = sorted(
                    {
                        item["label"]
                        for item in detections
                        if is_lego_target_label(item["label"])
                    }
                )
                if lego_labels and not is_lego_target_label(state["target_label"]):
                    selected_lego = choose_initial_lego_target(
                        state["target_label"],
                        lego_labels,
                    )
                    with state_lock:
                        runtime_state["target_label"] = selected_lego
                    state["target_label"] = selected_lego

            current_time = time.time()

            labels = sorted({item["label"] for item in detections})
            rescue_person_seen = enforce_rescue_person_stop(
                profile,
                labels,
                motion_controller,
            )
            if rescue_person_seen:
                if not previous_rescue_person_seen:
                    person_motion_status = motion_controller.status()
                    motion_controller.logger.record(
                        "person_detected",
                        position=list(person_motion_status["position"]),
                        labels=labels,
                    )
            previous_rescue_person_seen = rescue_person_seen

            help_signal_seen = HELP_SIGNAL_LABEL in labels
            if profile.navigation_policy in ("rescue", "lego"):
                help_signal_confirmed, new_help_signal = help_signal_gate.update(
                    help_signal_seen
                )
            else:
                help_signal_gate.reset()
                help_signal_confirmed = False
                new_help_signal = False

            if not help_signal_seen and not help_signal_confirmed:
                pending_help_check = False

            if new_help_signal:
                if motion_controller.armed:
                    motion_controller.disarm(
                        "semantic help signal; re-arm after assessment"
                    )
                help_motion_status = motion_controller.status()
                motion_controller.logger.record(
                    "lego_help_signal_confirmed",
                    pattern="red_cross",
                    position=list(help_motion_status["position"]),
                    labels=labels,
                )
                pending_help_check = True

            if (
                pending_help_check
                and state["auto_narration"]
                and not voice_busy.is_set()
            ):
                speak(
                    "Red LEGO cross help signal confirmed. Do you need help? "
                    "Please answer after the next beep."
                )
                request_voice_turn(state, labels)
                pending_help_check = False
                last_rescue_signature = (True, tuple(labels))
                last_rescue_announcement = current_time

            if (
                profile.navigation_policy == "rescue"
                and state["auto_narration"]
                and not voice_busy.is_set()
                and not help_signal_confirmed
                and not help_signal_seen
            ):
                rescue_signature = ("person" in labels, tuple(labels))
                if (
                    rescue_signature != last_rescue_signature
                    and current_time - last_rescue_announcement > 6
                ):
                    if "person" in labels:
                        rescue_message = (
                            "Person detected. Do you need help? "
                            "Please answer after the next beep."
                        )
                    elif labels:
                        rescue_message = (
                            "No person detected. I can see "
                            f"{format_spoken_list(labels)}."
                        )
                    else:
                        rescue_message = (
                            "No person detected. I am still scanning."
                        )

                    speak(rescue_message)
                    if "person" in labels:
                        request_voice_turn(state, labels)
                    last_rescue_signature = rescue_signature
                    last_rescue_announcement = current_time

            if (
                state["auto_narration"]
                and profile.navigation_policy != "rescue"
                and openai_client is not None
                and labels
                and not voice_busy.is_set()
                and current_time - last_ai_check > ai_interval
                and scene_requests.empty()
            ):
                last_ai_check = current_time
                ai_width = 960
                ai_height = round(ai_width * frame.shape[0] / frame.shape[1])
                ai_frame = cv2.resize(frame, (ai_width, ai_height))
                try:
                    scene_requests.put_nowait(
                        {
                            "frame": ai_frame,
                            "labels": labels,
                            "scenario_id": state["scenario_id"],
                            "target_label": state["target_label"],
                        }
                    )
                    print(f"[YOLO Trigger]: Found {', '.join(labels)}. Prompting OpenAI...")
                except queue.Full:
                    pass

            motion_status = motion_controller.status()
            navigation = navigator.plan(
                profile,
                detections,
                state["target_label"],
                frame.shape,
                motion_status,
                paused=state["mission_paused"] or dashboard.show_scenario_menu,
                interaction_active=(
                    help_signal_seen
                    or help_signal_confirmed
                    or rescue_person_seen
                    or voice_busy.is_set()
                ),
            )
            planned_motion = navigation.command
            if navigation.state == NavigationState.COMPLETE:
                motion_controller.disarm(
                    "mission complete; inspect result before re-arming"
                )
            motion_controller.apply(planned_motion, detections, frame.shape)
            motion_status = motion_controller.status()
            gimbal_command = gimbal_tracker.plan(
                navigation.selected_detection,
                frame.shape,
            )
            should_track_gimbal = (
                args.enable_gimbal_tracking
                and motion_controller.armed
                and navigation.selected_detection is not None
            )
            if should_track_gimbal:
                if motion_controller.apply_gimbal(
                    gimbal_command.pitch_dps,
                    gimbal_command.yaw_dps,
                ):
                    gimbal_active = bool(
                        gimbal_command.pitch_dps or gimbal_command.yaw_dps
                    )
                    gimbal_status = gimbal_command.reason
                else:
                    motion_status = motion_controller.status()
                    gimbal_active = False
                    gimbal_status = "failed"
                    dashboard.add_event("Gimbal tracking failed", "danger")
            else:
                if gimbal_active:
                    if not motion_controller.stop_gimbal():
                        motion_status = motion_controller.status()
                gimbal_active = False
                gimbal_status = (
                    "ready" if args.enable_gimbal_tracking else "disabled"
                )
            with state_lock:
                runtime_state["navigation_state"] = navigation.state.value
                runtime_state["navigation_reason"] = navigation.reason
                runtime_state["gimbal_status"] = gimbal_status
            if navigation.state != previous_navigation_state:
                dashboard.add_event(
                    f"Navigation: {navigation.state.value} - {navigation.reason}"
                )
                motion_controller.logger.record(
                    "navigation_state_changed",
                    scenario=profile.id,
                    state=navigation.state.value,
                    reason=navigation.reason,
                    target=state["target_label"],
                )
                if navigation.state == NavigationState.COMPLETE:
                    speak("Mission destination reached. Motion is disarmed.")
                previous_navigation_state = navigation.state

            selected_detection = navigation.selected_detection
            selected_box = (
                tuple(selected_detection["box"])
                if selected_detection is not None
                else None
            )
            for detection in detections:
                selected = (
                    profile.navigation_policy in ("target", "lego", "rescue")
                    and selected_box is not None
                    and detection["label"] == state["target_label"]
                    and tuple(detection["box"]) == selected_box
                )
                draw_detection(frame, detection, selected, annotation)

            if scenario_tracks_visual_target(state["scenario_id"]):
                guidance, _ = calculate_target_guidance(
                    [selected_detection] if selected_detection is not None else [],
                    state["target_label"],
                    frame.shape,
                )
                if selected_detection is not None:
                    x1, y1, x2, y2 = selected_detection["box"]
                    target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                    cv2.line(
                        frame,
                        frame_center,
                        target_center,
                        (0, 165, 255),
                        annotation["box_thickness"],
                        cv2.LINE_AA,
                    )
                if (
                    guidance != last_guidance
                    and current_time - last_guidance_spoken > 2.5
                    and not voice_busy.is_set()
                    and not state["mission_paused"]
                    and not dashboard.show_scenario_menu
                    and motion_controller.armed
                    and not help_signal_seen
                ):
                    speak(f"Target guidance: {guidance.lower()}.")
                    last_guidance = guidance
                    last_guidance_spoken = current_time
            else:
                last_guidance = None

            if args.enable_motion and current_time - last_position_log >= 1:
                motion_controller.logger.record(
                    "position",
                    position=list(motion_status["position"]),
                    motion_mode=motion_status["mode"],
                    motion_reason=motion_status["reason"],
                )
                last_position_log = current_time
            if current_time - last_map_save >= 5:
                try:
                    mission_map.save(args.mission_map)
                except Exception as error:
                    print(f"[Map]: Could not save mission map: {error}")
                last_map_save = current_time

            dashboard_state = get_runtime_state()
            connections = {
                "robot": bool(getattr(motion_backend, "connected", False)),
                "cloud": cloud_available.is_set(),
                "microphone": microphone_ready,
                "tts": tts_available.is_set(),
                "gimbal_status": dashboard_state["gimbal_status"],
            }
            preflight = {
                "tof": motion_status["sensor_ready"],
                "motion_opt_in": args.enable_motion,
                "cloud": connections["cloud"],
                "microphone": connections["microphone"],
                "tts": connections["tts"],
            }
            dashboard_frame = dashboard.render(
                frame,
                detections,
                dashboard_state,
                profile,
                tuple(scenario_catalog),
                motion_status,
                navigation,
                mission_map,
                lego_model_status,
                connections,
                preflight,
            )
            cv2.imshow(preview_window, dashboard_frame)
            preview_rendered = True

            key = cv2.waitKey(1) & 0xFF
            if not preview_window_is_visible(preview_window):
                motion_controller.emergency_stop("preview window closed")
                print("[Safety]: Preview window closed; motion stopped.")
                break
            global_talk_pressed = global_voice_hotkey_pressed()
            global_stop_pressed = global_emergency_stop_pressed()
            if key in (ord("q"), ord("Q")):
                break

            actions = dashboard.consume_actions()
            if key in (ord("e"), ord("E"), 27) or global_stop_pressed:
                actions.insert(0, {"type": "estop"})
            if key in (ord("r"), ord("R")):
                actions.append({"type": "reset_estop"})
            if key in (ord("m"), ord("M")):
                actions.append({"type": "arm_toggle"})
            if key in SCENARIO_KEYS:
                actions.append({"type": "scenario", "value": SCENARIO_KEYS[key]})
            if key in (ord("v"), ord("V"), ord(" ")) or global_talk_pressed:
                actions.append({"type": "talk"})
            if key in (ord("t"), ord("T")):
                actions.append({"type": "next_target"})
            if key in (ord("n"), ord("N")):
                actions.append({"type": "toggle_narration"})
            if key in (ord("p"), ord("P")):
                actions.append({"type": "toggle_pause"})
            if key in (ord("w"), ord("W")):
                actions.append({"type": "add_waypoint"})
            if key in (ord("h"), ord("H")):
                actions.append({"type": "return_home"})
            if key in (ord("l"), ord("L")):
                actions.append({"type": "read_objects"})
            if key in (ord("s"), ord("S")):
                dashboard.show_scenario_menu = True
                actions.append({"type": "open_scenarios"})

            for action in actions:
                action_type = action["type"]
                current_state = get_runtime_state()
                if action_type == "estop":
                    motion_controller.emergency_stop("operator emergency stop")
                    dashboard.add_event("EMERGENCY STOP activated", "danger")
                    play_beep(400, 300)
                    speak("Emergency stop activated.")
                elif action_type == "reset_estop":
                    if not dashboard.teacher_mode:
                        speak("Only teacher mode can reset the emergency stop.")
                    elif motion_controller.status()["mode"] != "ESTOP":
                        speak("The emergency stop is not latched.")
                    elif motion_controller.reset_emergency_stop():
                        dashboard.add_event("Emergency stop reset")
                        speak("Emergency stop reset. Motion remains disarmed.")
                    else:
                        dashboard.add_event("Emergency reset failed", "danger")
                        speak(
                            f"Emergency reset failed. {motion_controller.last_reason}."
                        )
                elif action_type == "arm_toggle":
                    current_profile = get_scenario_profile(
                        current_state["scenario_id"]
                    )
                    if motion_controller.armed:
                        motion_controller.disarm()
                        dashboard.add_event("Autonomous motion disarmed")
                        speak("Autonomous motion disarmed.")
                    elif not dashboard.teacher_mode:
                        speak("Only teacher mode can arm autonomous motion.")
                    elif current_state["mission_paused"] or dashboard.show_scenario_menu:
                        speak("Resume the mission and close the scenario menu before arming.")
                    elif not current_profile.allow_motion:
                        speak("This scenario is observation only and cannot be armed.")
                    elif motion_controller.arm():
                        dashboard.add_event("Autonomous motion armed", "danger")
                        speak("Autonomous motion armed at low speed.")
                    else:
                        speak(f"Motion not armed. {motion_controller.last_reason}.")
                elif action_type == "open_scenarios":
                    motion_controller.disarm("scenario menu opened; re-arm required")
                    with state_lock:
                        runtime_state["mission_paused"] = True
                    dashboard.add_event("Scenario menu opened")
                elif action_type == "scenario":
                    scenario_id = action["value"]
                    selected_profile = get_scenario_profile(scenario_id)
                    motion_controller.disarm("scenario changed; re-arm required")
                    target_label = selected_profile.preferred_target
                    if not target_label and selected_profile.target_labels:
                        target_label = selected_profile.target_labels[0]
                    if not target_label and selected_profile.navigation_policy == "lego":
                        lego_labels = [
                            label for label in labels if is_lego_target_label(label)
                        ]
                        target_label = lego_labels[0] if lego_labels else None
                    elif not target_label:
                        target_label = current_state["target_label"]
                    with state_lock:
                        runtime_state["scenario_id"] = scenario_id
                        runtime_state["target_label"] = target_label
                        runtime_state["mission_paused"] = False
                    navigator.select_target(target_label)
                    dashboard.show_scenario_menu = False
                    dashboard.add_event(f"Mission selected: {selected_profile.name}")
                    motion_controller.logger.record(
                        "scenario_selected",
                        scenario=scenario_id,
                        target=target_label,
                    )
                    last_guidance = None
                    speak(
                        f"Mission mode: {selected_profile.name}. "
                        f"Objective: {selected_profile.objective}"
                    )
                elif action_type == "toggle_pause":
                    paused = not current_state["mission_paused"]
                    if paused:
                        motion_controller.disarm("mission paused; re-arm required")
                    with state_lock:
                        runtime_state["mission_paused"] = paused
                    dashboard.add_event("Mission paused" if paused else "Mission resumed")
                    speak(
                        "Mission paused."
                        if paused
                        else "Mission resumed. Motion remains disarmed."
                    )
                elif action_type == "talk":
                    request_voice_turn(current_state, labels)
                elif action_type == "read_objects":
                    speak(f"I can currently see {format_spoken_list(labels)}.")
                elif action_type in ("next_target", "select_target"):
                    current_profile = get_scenario_profile(
                        current_state["scenario_id"]
                    )
                    if current_profile.navigation_policy not in (
                        "target",
                        "lego",
                        "rescue",
                    ):
                        speak("This mission does not use a selectable visual target.")
                        continue
                    target_choices = [
                        label
                        for label in labels
                        if profile_accepts_target(current_profile, label)
                    ]
                    if action_type == "select_target":
                        new_target = action["value"]
                        selected_item = action.get("detection")
                        if not profile_accepts_target(current_profile, new_target):
                            dashboard.add_event(
                                f"Target not allowed in this mission: {new_target}",
                                "danger",
                            )
                            speak(
                                f"{new_target} is not an allowed target in "
                                f"{current_profile.name}."
                            )
                            continue
                    else:
                        if not target_choices:
                            speak("No selectable targets are currently visible.")
                            continue
                        new_target = cycle_target(
                            current_state["target_label"],
                            target_choices,
                        )
                        selected_item = None
                    motion_controller.disarm("target changed; re-arm required")
                    with state_lock:
                        runtime_state["target_label"] = new_target
                    navigator.select_target(new_target, selected_item)
                    dashboard.add_event(f"Target selected: {new_target}")
                    motion_controller.logger.record(
                        "target_selected",
                        scenario=current_profile.id,
                        target=new_target,
                    )
                    last_guidance = None
                    speak(f"Selected target: {new_target}.")
                elif action_type == "add_waypoint":
                    if not dashboard.teacher_mode:
                        speak("Only teacher mode can record waypoints.")
                    else:
                        waypoint = mission_map.add_waypoint(
                            f"Waypoint {len(mission_map.waypoints) + 1}",
                            motion_status["position"],
                        )
                        motion_controller.logger.record(
                            "waypoint_recorded",
                            name=waypoint.name,
                            x=waypoint.x,
                            y=waypoint.y,
                        )
                        try:
                            mission_map.save(args.mission_map)
                        except Exception as error:
                            dashboard.add_event("Waypoint map save failed", "danger")
                            speak(f"Waypoint recorded in memory, but map save failed: {error}.")
                        else:
                            dashboard.add_event(f"Recorded {waypoint.name}")
                            speak(f"Recorded {waypoint.name}.")
                elif action_type == "return_home":
                    motion_controller.disarm("return-home selected; re-arm required")
                    if not dashboard.teacher_mode:
                        speak("Only teacher mode can start return to base.")
                    elif "return_home" not in scenario_catalog:
                        dashboard.add_event("Return-home profile unavailable", "danger")
                        speak("The loaded scenario file has no return-home profile.")
                    elif mission_map.home is None:
                        speak("No home position has been recorded yet.")
                    else:
                        with state_lock:
                            runtime_state["scenario_id"] = "return_home"
                            runtime_state["mission_paused"] = False
                        dashboard.show_scenario_menu = False
                        dashboard.add_event("Return-home mission selected")
                        motion_controller.logger.record(
                            "scenario_selected",
                            scenario="return_home",
                            target=None,
                        )
                        speak("Return to base selected. Check the map and re-arm manually.")
                elif action_type == "toggle_narration":
                    with state_lock:
                        runtime_state["auto_narration"] = not runtime_state[
                            "auto_narration"
                        ]
                        narration_enabled = runtime_state["auto_narration"]
                    speak(
                        "Automatic narration on."
                        if narration_enabled
                        else "Automatic narration off."
                    )
                elif action_type == "toggle_tts_mute":
                    with state_lock:
                        runtime_state["tts_muted"] = not runtime_state["tts_muted"]
                        muted = runtime_state["tts_muted"]
                    dashboard.add_event("TTS muted" if muted else "TTS unmuted")
                    if not muted:
                        speak("Text to speech unmuted.")
                elif action_type == "tts_rate":
                    with state_lock:
                        runtime_state["tts_rate"] = max(
                            90,
                            min(220, runtime_state["tts_rate"] + action["value"]),
                        )
                        rate = runtime_state["tts_rate"]
                    dashboard.add_event(f"TTS rate: {rate}")
                    speak(f"Speech rate {rate}.")
                elif action_type == "tts_volume":
                    with state_lock:
                        runtime_state["tts_volume"] = max(
                            0.0,
                            min(
                                1.0,
                                runtime_state["tts_volume"] + action["value"],
                            ),
                        )
                        volume = runtime_state["tts_volume"]
                    dashboard.add_event(f"TTS volume: {round(volume * 100)}%")
                    speak(f"Speech volume {round(volume * 100)} percent.")
                elif action_type == "toggle_role":
                    with state_lock:
                        runtime_state["ui_role"] = action["value"]
                    if action["value"] == "student" and motion_controller.armed:
                        motion_controller.disarm("student mode selected; re-arm required")
                    dashboard.add_event(f"UI role: {action['value']}")
    finally:
        shutdown_event.set()
        try:
            motion_controller.emergency_stop("application shutdown")
        except Exception as error:
            print(f"[Safety]: Emergency shutdown command failed: {error}")
        try:
            motion_controller.close()
        except Exception as error:
            print(f"[Safety]: Motion backend cleanup failed: {error}")
        try:
            atexit.unregister(motion_controller.close)
        except Exception:
            pass
        try:
            mission_map.save(args.mission_map)
        except Exception as error:
            print(f"[Map]: Could not save mission map during shutdown: {error}")
        if lego_detector is not None:
            try:
                lego_detector.close()
            except Exception as error:
                print(f"[LEGO Model]: Cleanup failed: {error}")
        try:
            sd.stop()
        except Exception:
            pass
        for work_queue in (voice_requests, scene_requests, speech_queue):
            try:
                work_queue.put_nowait(None)
            except queue.Full:
                pass
        for worker in workers:
            worker.join(timeout=1)
        if openai_client is not None:
            try:
                openai_client.close()
            except Exception:
                pass
        try:
            sct.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        for worker in workers:
            worker.join(timeout=2)


if __name__ == "__main__":
    main()
