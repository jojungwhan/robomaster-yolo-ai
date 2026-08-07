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
    MotionCommand,
    RoboMasterBackend,
    SafetyMotionController,
)
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

SCENARIOS = {
    "exploration": {
        "name": "Exploration",
        "system_prompt": (
            "You are the voice of a DJI RoboMaster S1 exploration assistant. "
            "Answer naturally, remember the conversation, and keep spoken replies "
            "to two short sentences unless the user asks for more detail."
        ),
        "scene_prompt": (
            "Give one short, useful, slightly witty observation about the scene."
        ),
    },
    "rescue": {
        "name": "Search & Rescue (SDG + Alpha)",
        "system_prompt": (
            "You are a cautious search-and-rescue training assistant for disaster "
            "response, supporting SDG 3 and SDG 11 goals. Prioritize possible people, "
            "hazards, exits, and useful supplies. Treat computer-vision labels as "
            "unverified and never claim that someone is safe or that emergency help "
            "has been contacted. Keep spoken replies short and actionable."
        ),
        "scene_prompt": (
            "Give one concise search-and-rescue status report. Mention possible people, "
            "hazards, or useful supplies, and clearly express uncertainty."
        ),
    },
    "target": {
        "name": "Target Guidance",
        "system_prompt": (
            "You are a visual target-guidance assistant for a DJI RoboMaster S1. "
            "Motion may be enabled through a range-sensor safety controller. Never "
            "claim movement succeeded unless the motion status confirms it."
        ),
        "scene_prompt": (
            "Briefly report the selected target and any useful visual context. "
            "Do not claim that the robot moved."
        ),
    },
    "lego": {
        "name": "LEGO Search",
        "system_prompt": (
            "You are a LEGO search assistant. Identify colored LEGO candidates, "
            "Lego-Identification piece classes, semantic LEGO patterns, and "
            "ArUco-tagged builds. Explain uncertainty and keep replies concise."
        ),
        "scene_prompt": (
            "Briefly name the LEGO pieces, semantic patterns, or marker IDs and report "
            "the selected target. A red LEGO cross means help is requested."
        ),
    },
}

SCENARIO_KEYS = {
    ord("1"): "exploration",
    ord("2"): "rescue",
    ord("3"): "target",
    ord("4"): "lego",
}

shutdown_event = threading.Event()
voice_busy = threading.Event()
tts_speaking = threading.Event()
speech_queue = queue.Queue()
voice_requests = queue.Queue(maxsize=1)
scene_requests = queue.Queue(maxsize=1)
state_lock = threading.Lock()

runtime_state = {
    "scenario_id": "exploration",
    "target_label": "person",
    "auto_narration": True,
    "voice_status": "Press V, Space, or F8 to talk",
}

conversation_histories = {scenario_id: [] for scenario_id in SCENARIOS}


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
        speech_queue.put(text)


def speech_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
        print("[TTS]: Speech engine ready.")
    except Exception as error:
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
            engine.say(text)
            engine.runAndWait()
        except Exception as error:
            print(f"Text-to-speech error: {error}")
        finally:
            tts_speaking.clear()

    engine.stop()


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

    if scenario_id in ("target", "lego"):
        prompt += f" The selected visual target is {target_label or 'not selected'}."

    return prompt


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
                tts_speaking.is_set() or not speech_queue.empty()
            ) and not shutdown_event.is_set() and time.monotonic() < speech_wait_deadline:
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
            set_voice_status("Thinking...")

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
                conversation_histories[scenario_id] = conversation_histories[
                    scenario_id
                ][-MAX_HISTORY_MESSAGES:]

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
                if scenario_id in ("target", "lego")
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


def draw_status_panel(
    frame,
    state,
    detections,
    guidance,
    annotation_scale,
    motion_status,
    lego_model_status,
):
    labels = sorted({item["label"] for item in detections})
    label_summary = ", ".join(labels) if labels else "none"
    if len(label_summary) > 65:
        label_summary = label_summary[:62] + "..."

    lines = [
        (
            f"Mission: {SCENARIOS[state['scenario_id']]['name']} | "
            f"Narration: {'ON' if state['auto_narration'] else 'OFF'}",
            (255, 255, 255),
        ),
        (f"CV: {len(detections)} | {label_summary}", (0, 255, 0)),
    ]

    ranges = motion_status["ranges"]
    range_text = " ".join(
        f"{name[0].upper()}:{distance:.0f}"
        for name, distance in ranges.items()
    ) or "no ToF"
    position = motion_status["position"]
    motion_color = {
        "ARMED": (0, 255, 0),
        "ESTOP": (0, 0, 255),
    }.get(motion_status["mode"], (0, 165, 255))
    lines.append(
        (
            f"Motion: {motion_status['mode']} ({motion_status['backend']}) | "
            f"{motion_status['reason']} | ToF mm {range_text} | "
            f"pos {position[0]:.2f},{position[1]:.2f},{position[2]:.0f}",
            motion_color,
        )
    )

    if state["scenario_id"] in ("target", "lego"):
        lines.append(
            (
                f"Target: {state['target_label']} | {guidance}",
                (0, 165, 255),
            )
        )

    if state["scenario_id"] in ("rescue", "lego"):
        lines.append((f"Lego-Identification: {lego_model_status}", (180, 220, 255)))

    lines.extend(
        [
            (f"Voice: {state['voice_status']}", (255, 255, 0)),
            (
                "Keys: 1 Explore | 2 Rescue | 3 Target | 4 LEGO | "
                "V/Space/F8 Talk | T Next target",
                (220, 220, 220),
            ),
            (
                "Motion: M Arm/Disarm | E/Esc EMERGENCY STOP | R Reset | "
                "N Narration | Q Quit",
                (220, 220, 220),
            ),
        ]
    )

    row_height = 17
    panel_height = round((8 + row_height * len(lines)) * annotation_scale)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)

    for index, (text, color) in enumerate(lines):
        y = round((16 + row_height * index) * annotation_scale)
        cv2.putText(
            frame,
            text,
            (round(8 * annotation_scale), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38 * annotation_scale,
            color,
            max(1, round(0.8 * annotation_scale)),
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
    return label.startswith("lego_") and not label.startswith("lego_signal_")


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
    args = parse_args(argv)
    tof_layout = tuple(
        item.strip().lower() for item in args.tof_layout.split(",") if item.strip()
    )
    if "front" not in tof_layout:
        raise SystemExit("--tof-layout must include a front sensor.")
    if args.min_tof_count < 1 or args.min_tof_count > len(tof_layout):
        raise SystemExit("--min-tof-count must fit the configured --tof-layout.")

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
    yolo_model = YOLO("yolov8n.pt")
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
    openai_client = OpenAI()

    sct = mss.MSS()
    if args.screen < 1 or args.screen >= len(sct.monitors):
        motion_controller.close()
        raise SystemExit(f"--screen must be between 1 and {len(sct.monitors) - 1}.")
    monitor = dict(sct.monitors[args.screen])

    preview_window = "RoboMaster S1 - YOLO CV Pipeline"
    preview_width = 640
    preview_height = round(preview_width * monitor["height"] / monitor["width"])
    preview_x = monitor["left"] + monitor["width"] - preview_width - 30
    preview_y = monitor["top"] + 30

    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview_window, preview_width, preview_height)
    cv2.moveWindow(preview_window, preview_x, preview_y)

    try:
        cv2.setWindowProperty(preview_window, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        print("OpenCV could not enable always-on-top mode on this system.")

    exclude_preview_from_capture(preview_window)

    annotation_scale = max(
        monitor["width"] / preview_width,
        monitor["height"] / preview_height,
    )
    annotation = {
        "box_thickness": max(2, round(2 * annotation_scale)),
        "font_scale": 0.55 * annotation_scale,
        "font_thickness": max(1, round(annotation_scale)),
        "label_padding": max(2, round(4 * annotation_scale)),
    }

    workers = [
        threading.Thread(target=speech_worker, name="speech", daemon=True),
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
    for worker in workers:
        worker.start()

    print("YOLO + OpenAI Mission System Active.")
    print(
        "Keys: 1 Explore | 2 Rescue | 3 Target | 4 LEGO | "
        "V/Space/F8 Talk | T Next target | M Arm | E/Esc Stop | R Reset | Q Quit"
    )
    print(
        f"[Motion]: backend={args.motion_backend}, requested={args.enable_motion}, "
        f"ToF layout={','.join(tof_layout)}. Motion starts DISARMED."
    )
    speak(
        "Mission system online. Press one, two, three, or four to select a scenario. "
        "Press V, Space, or F8 to talk."
    )

    last_ai_check = time.time()
    ai_interval = 12
    detection_confidence = 0.20
    inference_size = 640
    last_guidance = None
    last_guidance_spoken = 0
    last_rescue_signature = None
    last_rescue_announcement = 0
    previous_scenario = "exploration"
    last_position_log = 0
    help_signal_gate = ConsecutiveDetectionGate(required_frames=3, release_frames=8)
    pending_help_check = False

    try:
        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            state = get_runtime_state()
            if state["scenario_id"] != previous_scenario:
                previous_scenario = state["scenario_id"]
                last_rescue_signature = None
                last_rescue_announcement = 0
                help_signal_gate.reset()
                pending_help_check = False

            results = yolo_model(
                frame,
                stream=True,
                verbose=False,
                conf=detection_confidence,
                imgsz=inference_size,
            )
            detections = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    detection = {
                        "box": (x1, y1, x2, y2),
                        "label": yolo_model.names[class_id],
                        "confidence": float(box.conf[0]),
                        "source": "yolo",
                    }
                    detections.append(detection)

            if state["scenario_id"] in ("rescue", "lego"):
                if lego_detector is not None:
                    try:
                        detections.extend(lego_detector.detect(frame))
                    except Exception as error:
                        lego_model_status = f"OFF ({type(error).__name__})"
                        print(f"[LEGO Model]: Inference failed; disabled: {error}")
                        lego_detector.close()
                        lego_detector = None

                if state["scenario_id"] == "lego":
                    detections.extend(detect_lego(frame))
                else:
                    detections.extend(detect_red_cross_signal(frame))

            if state["scenario_id"] == "lego":

                lego_labels = sorted(
                    {
                        item["label"]
                        for item in detections
                        if is_lego_target_label(item["label"])
                    }
                )
                if lego_labels and state["target_label"] not in lego_labels:
                    marker_labels = [
                        label for label in lego_labels if label.startswith("lego_marker_")
                    ]
                    selected_lego = marker_labels[0] if marker_labels else lego_labels[0]
                    with state_lock:
                        runtime_state["target_label"] = selected_lego
                    state["target_label"] = selected_lego

            for detection in detections:
                selected = (
                    state["scenario_id"] in ("target", "lego")
                    and detection["label"] == state["target_label"]
                )
                draw_detection(frame, detection, selected, annotation)

            guidance = ""
            selected_detection = None
            current_time = time.time()

            if state["scenario_id"] in ("target", "lego"):
                guidance, selected_detection = calculate_target_guidance(
                    detections,
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
                    and not any(
                        item["label"] == HELP_SIGNAL_LABEL for item in detections
                    )
                ):
                    speak(f"Target guidance: {guidance.lower()}.")
                    last_guidance = guidance
                    last_guidance_spoken = current_time
            else:
                last_guidance = None

            labels = sorted({item["label"] for item in detections})
            help_signal_seen = HELP_SIGNAL_LABEL in labels
            if state["scenario_id"] in ("rescue", "lego"):
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
                state["scenario_id"] == "rescue"
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
                        if motion_controller.armed:
                            motion_controller.disarm(
                                "person interaction; re-arm after conversation"
                            )
                        motion_status = motion_controller.status()
                        motion_controller.logger.record(
                            "person_detected",
                            position=list(motion_status["position"]),
                            labels=labels,
                        )
                        request_voice_turn(state, labels)
                    last_rescue_signature = rescue_signature
                    last_rescue_announcement = current_time

            if (
                state["auto_narration"]
                and state["scenario_id"] != "rescue"
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
            if help_signal_seen or help_signal_confirmed:
                planned_motion = MotionCommand(reason="semantic help signal stop")
            elif voice_busy.is_set():
                planned_motion = MotionCommand(reason="voice interaction pause")
            else:
                planned_motion = planner.plan(
                    state["scenario_id"],
                    detections,
                    state["target_label"],
                    frame.shape,
                    motion_status["ranges"],
                )
            motion_controller.apply(planned_motion, detections, frame.shape)
            motion_status = motion_controller.status()

            if args.enable_motion and current_time - last_position_log >= 1:
                motion_controller.logger.record(
                    "position",
                    position=list(motion_status["position"]),
                    motion_mode=motion_status["mode"],
                    motion_reason=motion_status["reason"],
                )
                last_position_log = current_time

            draw_status_panel(
                frame,
                get_runtime_state(),
                detections,
                guidance,
                annotation_scale,
                motion_status,
                lego_model_status,
            )
            cv2.imshow(preview_window, frame)

            key = cv2.waitKey(1) & 0xFF
            global_talk_pressed = global_voice_hotkey_pressed()
            global_stop_pressed = global_emergency_stop_pressed()

            if key in (ord("e"), ord("E"), 27) or global_stop_pressed:
                motion_controller.emergency_stop("operator emergency stop")
                play_beep(400, 300)
                speak("Emergency stop activated.")
                continue

            if key in (ord("r"), ord("R")):
                motion_controller.reset_emergency_stop()
                speak("Emergency stop reset. Motion remains disarmed.")
                continue

            if key in (ord("m"), ord("M")):
                if motion_controller.armed:
                    motion_controller.disarm()
                    speak("Autonomous motion disarmed.")
                elif motion_controller.arm():
                    speak("Autonomous motion armed at low speed.")
                else:
                    speak(f"Motion not armed. {motion_controller.last_reason}.")
                continue

            if key in (ord("q"), ord("Q")):
                break

            if key in SCENARIO_KEYS:
                scenario_id = SCENARIO_KEYS[key]
                if motion_controller.armed:
                    motion_controller.disarm("scenario changed; re-arm required")
                with state_lock:
                    runtime_state["scenario_id"] = scenario_id
                    if scenario_id == "target" and labels:
                        runtime_state["target_label"] = (
                            "person" if "person" in labels else labels[0]
                        )
                    elif scenario_id == "lego":
                        lego_labels = [
                            label for label in labels if is_lego_target_label(label)
                        ]
                        if lego_labels:
                            runtime_state["target_label"] = lego_labels[0]
                last_guidance = None
                speak(f"Mission mode: {SCENARIOS[scenario_id]['name']}.")
                continue

            if key in (ord("v"), ord("V"), ord(" ")) or global_talk_pressed:
                request_voice_turn(state, labels)
                continue

            if key in (ord("t"), ord("T")):
                target_choices = labels
                if state["scenario_id"] == "lego":
                    target_choices = [
                        label for label in labels if is_lego_target_label(label)
                    ]
                new_target = cycle_target(state["target_label"], target_choices)
                if motion_controller.armed:
                    motion_controller.disarm("target changed; re-arm required")
                with state_lock:
                    runtime_state["target_label"] = new_target
                last_guidance = None
                speak(f"Selected target: {new_target}.")
                continue

            if key in (ord("n"), ord("N")):
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
    finally:
        if lego_detector is not None:
            lego_detector.close()
        motion_controller.emergency_stop("application shutdown")
        motion_controller.close()
        atexit.unregister(motion_controller.close)
        shutdown_event.set()
        sd.stop()
        for work_queue in (voice_requests, scene_requests, speech_queue):
            try:
                work_queue.put_nowait(None)
            except queue.Full:
                pass

        sct.close()
        cv2.destroyAllWindows()
        for worker in workers:
            worker.join(timeout=2)


if __name__ == "__main__":
    main()
