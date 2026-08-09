"""OpenCV classroom dashboard with mouse controls and mission telemetry."""

from collections import deque
import queue
import textwrap
import time
import math

import cv2
import numpy as np


COLORS = {
    "background": (20, 24, 32),
    "panel": (31, 38, 49),
    "panel_light": (44, 53, 68),
    "text": (240, 243, 247),
    "muted": (165, 175, 190),
    "green": (70, 210, 120),
    "amber": (0, 184, 255),
    "red": (60, 60, 235),
    "blue": (230, 150, 55),
    "cyan": (220, 210, 70),
}


def _point_in_rect(point, rect):
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _fit_rect(source_shape, destination):
    source_height, source_width = source_shape[:2]
    x1, y1, x2, y2 = destination
    available_width = x2 - x1
    available_height = y2 - y1
    scale = min(available_width / source_width, available_height / source_height)
    width = max(1, round(source_width * scale))
    height = max(1, round(source_height * scale))
    left = x1 + (available_width - width) // 2
    top = y1 + (available_height - height) // 2
    return (left, top, left + width, top + height), scale


class ClassroomDashboard:
    def __init__(self, width=1280, height=840):
        self.width = width
        self.height = height
        self.sidebar_width = 360
        self.header_height = 52
        self.caption_height = 118
        self.show_scenario_menu = True
        self.teacher_mode = True
        self._actions = queue.Queue()
        self._hit_regions = []
        self._detection_regions = []
        self.events = deque(maxlen=7)
        self.last_render_metadata = {}

    @property
    def hit_regions(self):
        return list(self._hit_regions)

    def add_event(self, text, level="info"):
        self.events.appendleft(
            {
                "time": time.strftime("%H:%M:%S"),
                "text": str(text),
                "level": level,
            }
        )

    def mouse_callback(self, event, x, y, _flags, _userdata=None):
        if event != cv2.EVENT_LBUTTONUP:
            return
        for region in reversed(self._hit_regions):
            if _point_in_rect((x, y), region["rect"]):
                action = dict(region["action"])
                if action["type"] == "open_scenarios":
                    self.show_scenario_menu = True
                elif action["type"] == "scenario":
                    self.show_scenario_menu = False
                elif action["type"] == "toggle_role":
                    self.teacher_mode = not self.teacher_mode
                    action["value"] = "teacher" if self.teacher_mode else "student"
                self._actions.put(action)
                return

        if not self.show_scenario_menu:
            for region in reversed(self._detection_regions):
                if _point_in_rect((x, y), region["rect"]):
                    self._actions.put(
                        {
                            "type": "select_target",
                            "value": region["label"],
                            "detection": region["detection"],
                        }
                    )
                    return

    def consume_actions(self):
        actions = []
        while True:
            try:
                actions.append(self._actions.get_nowait())
            except queue.Empty:
                return actions

    def _text(
        self,
        canvas,
        text,
        origin,
        scale=0.45,
        color=None,
        thickness=1,
    ):
        cv2.putText(
            canvas,
            str(text),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color or COLORS["text"],
            thickness,
            cv2.LINE_AA,
        )

    def _wrapped_text(
        self,
        canvas,
        text,
        origin,
        width_chars,
        scale=0.42,
        color=None,
        line_height=18,
        max_lines=3,
    ):
        x, y = origin
        lines = textwrap.wrap(str(text), width=max(8, width_chars)) or [""]
        for index, line in enumerate(lines[:max_lines]):
            self._text(
                canvas,
                line,
                (x, y + index * line_height),
                scale=scale,
                color=color,
            )
        return min(len(lines), max_lines) * line_height

    def _button(self, canvas, rect, label, action, color=None, enabled=True):
        fill = color or COLORS["panel_light"]
        if not enabled:
            fill = (45, 45, 45)
        cv2.rectangle(canvas, rect[:2], rect[2:], fill, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, rect[:2], rect[2:], (90, 100, 116), 1, cv2.LINE_AA)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        text_x = rect[0] + max(4, (rect[2] - rect[0] - text_size[0]) // 2)
        text_y = rect[1] + (rect[3] - rect[1] + text_size[1]) // 2
        self._text(
            canvas,
            label,
            (text_x, text_y),
            scale=0.38,
            color=COLORS["text"] if enabled else COLORS["muted"],
        )
        if enabled:
            self._hit_regions.append({"rect": rect, "action": action})

    def _draw_header(
        self,
        canvas,
        state,
        profile,
        motion_status,
        navigation,
        connections,
    ):
        cv2.rectangle(
            canvas,
            (0, 0),
            (self.width, self.header_height),
            COLORS["panel"],
            -1,
        )
        self._text(canvas, "ROBOMASTER CLASSROOM MISSION", (16, 22), 0.52, COLORS["cyan"], 1)
        motion_mode = motion_status.get("mode", "DISARMED")
        motion_color = {
            "ARMED": COLORS["green"],
            "ESTOP": COLORS["red"],
        }.get(motion_mode, COLORS["amber"])
        cv2.rectangle(canvas, (288, 7), (382, 30), motion_color, -1, cv2.LINE_AA)
        self._text(
            canvas,
            motion_mode,
            (299, 23),
            0.38,
            (15, 20, 26),
            1,
        )
        self._text(
            canvas,
            (
                f"{profile.name}  |  NAV: {navigation.state.value}"
                f"  |  TARGET: {state.get('target_label') or 'none'}"
                f"  |  GIMBAL: {connections.get('gimbal_status', 'disabled').upper()}"
            )[:104],
            (16, 43),
            0.43,
        )
        indicators = [
            ("ROBOT", connections.get("robot", False)),
            ("ToF", motion_status.get("sensor_ready", False)),
            ("CLOUD", connections.get("cloud", False)),
            ("MIC", connections.get("microphone", False)),
            ("TTS", connections.get("tts", False)),
        ]
        x = self.width - 330
        for label, ready in indicators:
            color = COLORS["green"] if ready else COLORS["red"]
            cv2.circle(canvas, (x, 21), 5, color, -1, cv2.LINE_AA)
            self._text(canvas, label, (x + 9, 25), 0.34, COLORS["muted"])
            x += 64

    def _draw_video(self, canvas, camera_frame, detections, target_selectable):
        video_area = (
            0,
            self.header_height,
            self.width - self.sidebar_width,
            self.height - self.caption_height,
        )
        display_rect, scale = _fit_rect(camera_frame.shape, video_area)
        left, top, right, bottom = display_rect
        resized = cv2.resize(camera_frame, (right - left, bottom - top))
        canvas[top:bottom, left:right] = resized

        self._detection_regions = []
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            mapped = (
                round(left + x1 * scale),
                round(top + y1 * scale),
                round(left + x2 * scale),
                round(top + y2 * scale),
            )
            if target_selectable:
                self._detection_regions.append(
                    {
                        "rect": mapped,
                        "label": detection["label"],
                        "detection": dict(detection),
                    }
                )

        self.last_render_metadata["video_rect"] = display_rect

    def _draw_controls(self, canvas, state, motion_status, target_selectable):
        left = self.width - self.sidebar_width + 12
        width = self.sidebar_width - 24
        y = self.header_height + 12
        role = "TEACHER" if self.teacher_mode else "STUDENT"
        self._button(
            canvas,
            (left, y, left + 104, y + 32),
            "SCENARIOS",
            {"type": "open_scenarios"},
            COLORS["blue"],
        )
        self._button(
            canvas,
            (left + 112, y, left + 216, y + 32),
            role,
            {"type": "toggle_role"},
        )
        self._button(
            canvas,
            (left + 224, y, left + width, y + 32),
            "TALK",
            {"type": "talk"},
            COLORS["blue"],
        )
        y += 40
        self._button(
            canvas,
            (left, y, left + 104, y + 32),
            "READ OBJECTS",
            {"type": "read_objects"},
        )
        self._button(
            canvas,
            (left + 112, y, left + 216, y + 32),
            "PAUSE" if not state.get("mission_paused") else "RESUME",
            {"type": "toggle_pause"},
            COLORS["amber"],
        )
        self._button(
            canvas,
            (left + 224, y, left + width, y + 32),
            "ESTOP",
            {"type": "estop"},
            COLORS["red"],
        )
        y += 40
        small_gap = 5
        small_width = (width - small_gap * 4) // 5
        tts_actions = (
            (
                "UNMUTE" if state.get("tts_muted") else "MUTE",
                {"type": "toggle_tts_mute"},
            ),
            ("RATE-", {"type": "tts_rate", "value": -10}),
            ("RATE+", {"type": "tts_rate", "value": 10}),
            ("VOL-", {"type": "tts_volume", "value": -0.1}),
            ("VOL+", {"type": "tts_volume", "value": 0.1}),
        )
        for index, (label, action) in enumerate(tts_actions):
            x1 = left + index * (small_width + small_gap)
            self._button(
                canvas,
                (x1, y, x1 + small_width, y + 28),
                label,
                action,
            )
        y += 36
        if self.teacher_mode:
            arm_label = "DISARM" if motion_status["mode"] == "ARMED" else "ARM"
            self._button(
                canvas,
                (left, y, left + 104, y + 32),
                arm_label,
                {"type": "arm_toggle"},
                COLORS["green"] if arm_label == "ARM" else COLORS["amber"],
                enabled=motion_status["mode"] != "ESTOP",
            )
            self._button(
                canvas,
                (left + 112, y, left + 216, y + 32),
                "RESET",
                {"type": "reset_estop"},
                enabled=motion_status["mode"] == "ESTOP",
            )
            self._button(
                canvas,
                (left + 224, y, left + width, y + 32),
                "ADD WAYPOINT",
                {"type": "add_waypoint"},
            )
            y += 40
            self._button(
                canvas,
                (left, y, left + 160, y + 32),
                "RETURN HOME",
                {"type": "return_home"},
            )
            self._button(
                canvas,
                (left + 168, y, left + width, y + 32),
                "NEXT TARGET",
                {"type": "next_target"},
                enabled=target_selectable,
            )
            y += 42
        self._text(
            canvas,
            f"STATUS: {motion_status.get('reason', 'unknown')}"[:48],
            (left, y + 12),
            0.30,
            COLORS["muted"],
        )
        y += 20
        return y

    def _draw_tof(self, canvas, ranges, top):
        left = self.width - self.sidebar_width + 12
        center_x = left + 82
        center_y = top + 67
        self._text(canvas, "ToF CLEARANCE (mm)", (left, top + 14), 0.39, COLORS["cyan"])
        cv2.rectangle(canvas, (center_x - 18, center_y - 28), (center_x + 18, center_y + 28), COLORS["blue"], -1)
        positions = {
            "front": (center_x - 22, center_y - 42),
            "rear": (center_x - 22, center_y + 52),
            "left": (center_x - 76, center_y + 5),
            "right": (center_x + 34, center_y + 5),
        }
        for direction, origin in positions.items():
            distance = ranges.get(direction)
            color = COLORS["green"] if distance and distance >= 700 else COLORS["red"]
            text = f"{direction[0].upper()} {distance:.0f}" if distance else f"{direction[0].upper()} --"
            self._text(canvas, text, origin, 0.35, color)
        return top + 125

    def _draw_objects(self, canvas, detections, top):
        left = self.width - self.sidebar_width + 12
        self._text(canvas, "DETECTED OBJECTS (click video to select)", (left, top + 14), 0.37, COLORS["cyan"])
        grouped = {}
        for item in detections:
            label = item["label"]
            grouped.setdefault(label, []).append(item["confidence"])
        y = top + 34
        for label, confidences in sorted(grouped.items())[:8]:
            summary = f"{label}  x{len(confidences)}  {max(confidences):.2f}"
            self._text(canvas, summary[:43], (left, y), 0.35, COLORS["text"])
            y += 18
        if not grouped:
            self._text(canvas, "none", (left, y), 0.36, COLORS["muted"])
            y += 18
        return max(top + 62, y + 8)

    def _draw_map(self, canvas, mission_map, position, top):
        left = self.width - self.sidebar_width + 12
        right = self.width - 12
        bottom = min(self.height - 108, top + 150)
        cv2.rectangle(canvas, (left, top), (right, bottom), (17, 21, 28), -1)
        self._text(canvas, "ODOMETRY / WAYPOINT MAP", (left + 8, top + 16), 0.36, COLORS["cyan"])
        points = list(mission_map.trail) + list(mission_map.obstacles)
        points.extend((wp.x, wp.y) for wp in mission_map.waypoints)
        current_x, current_y = position[:2]
        points.append((current_x, current_y))
        if mission_map.home:
            points.append(mission_map.home)
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            plot_left, plot_top = left + 8, top + 24
            plot_width, plot_height = right - left - 16, bottom - top - 32

            def map_point(point):
                px = plot_left + int((point[0] - min(xs)) / span * plot_width)
                py = plot_top + plot_height - int((point[1] - min(ys)) / span * plot_height)
                return px, py

            trail = [map_point(point) for point in mission_map.trail]
            for first, second in zip(trail, trail[1:]):
                cv2.line(canvas, first, second, COLORS["blue"], 1, cv2.LINE_AA)
            for obstacle in list(mission_map.obstacles)[-120:]:
                cv2.circle(canvas, map_point(obstacle), 1, COLORS["red"], -1)
            if mission_map.home:
                cv2.circle(canvas, map_point(mission_map.home), 5, COLORS["green"], 1)
            for waypoint in mission_map.waypoints:
                cv2.circle(canvas, map_point((waypoint.x, waypoint.y)), 4, COLORS["amber"], -1)
            cv2.circle(canvas, map_point((current_x, current_y)), 4, COLORS["cyan"], -1)
        return bottom + 8

    def _draw_events(self, canvas, top):
        left = self.width - self.sidebar_width + 12
        self._text(canvas, "EVENT TIMELINE", (left, top + 14), 0.36, COLORS["cyan"])
        y = top + 34
        for event in list(self.events)[:5]:
            color = COLORS["red"] if event["level"] == "danger" else COLORS["muted"]
            self._text(canvas, f"{event['time']} {event['text']}"[:47], (left, y), 0.32, color)
            y += 17

    def _draw_captions(self, canvas, state):
        top = self.height - self.caption_height
        right = self.width - self.sidebar_width
        cv2.rectangle(canvas, (0, top), (right, self.height), COLORS["panel"], -1)
        self._text(canvas, "STUDENT", (14, top + 23), 0.38, COLORS["blue"])
        self._wrapped_text(
            canvas,
            state.get("student_caption") or "Press TALK, V, Space, or F8 to speak.",
            (96, top + 23),
            96,
            0.43,
            COLORS["text"],
            19,
            2,
        )
        self._text(canvas, "ROBOT", (14, top + 72), 0.38, COLORS["green"])
        self._wrapped_text(
            canvas,
            state.get("robot_caption") or "Waiting for mission selection.",
            (96, top + 72),
            96,
            0.43,
            COLORS["text"],
            19,
            2,
        )
        self._text(
            canvas,
            f"VOICE: {state.get('voice_status', 'idle')}"[:116],
            (14, self.height - 7),
            0.29,
            COLORS["muted"],
        )

    def _draw_scenario_menu(self, canvas, profiles, current_id, preflight):
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)
        self._text(canvas, "SELECT A CLASSROOM MISSION", (32, 42), 0.76, COLORS["cyan"], 2)
        self._text(
            canvas,
            "Selecting a mission always disarms motion. Teacher must complete preflight and arm manually.",
            (32, 68),
            0.42,
            COLORS["muted"],
        )

        checks = [
            ("ToF", preflight.get("tof", False)),
            ("Motion opt-in", preflight.get("motion_opt_in", False)),
            ("Cloud", preflight.get("cloud", False)),
            ("Microphone", preflight.get("microphone", False)),
            ("TTS", preflight.get("tts", False)),
        ]
        x = 32
        for label, ready in checks:
            color = COLORS["green"] if ready else COLORS["red"]
            cv2.circle(canvas, (x, 91), 5, color, -1)
            self._text(canvas, label, (x + 10, 95), 0.35, COLORS["muted"])
            x += 132

        columns = 3
        gap = 14
        card_width = (self.width - 64 - gap * (columns - 1)) // columns
        start_y = 116
        rows = max(1, math.ceil(len(profiles) / columns))
        card_height = max(
            104,
            (self.height - start_y - 18 - gap * (rows - 1)) // rows,
        )
        compact = card_height < 145
        self._hit_regions = []
        for index, profile in enumerate(profiles):
            row, column = divmod(index, columns)
            x1 = 32 + column * (card_width + gap)
            y1 = start_y + row * (card_height + gap)
            x2 = x1 + card_width
            y2 = y1 + card_height
            selected = profile.id == current_id
            fill = (55, 70, 88) if selected else COLORS["panel"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                COLORS["cyan"] if selected else (80, 92, 108),
                2 if selected else 1,
                cv2.LINE_AA,
            )
            key = (
                f"[{profile.key}] "
                if profile.key
                else "[H] " if profile.id == "return_home" else ""
            )
            self._text(
                canvas,
                key + profile.short_name,
                (x1 + 10, y1 + (21 if compact else 25)),
                0.46 if compact else 0.54,
                COLORS["cyan"],
                1,
            )
            self._text(
                canvas,
                profile.name[:42],
                (x1 + 10, y1 + (41 if compact else 49)),
                0.34 if compact else 0.39,
                COLORS["text"],
            )
            self._text(
                canvas,
                f"Difficulty: {profile.difficulty}",
                (x1 + 10, y1 + (60 if compact else 72)),
                0.31 if compact else 0.34,
                COLORS["amber"],
            )
            self._wrapped_text(
                canvas,
                profile.objective,
                (x1 + 10, y1 + (79 if compact else 96)),
                46,
                0.31 if compact else 0.36,
                COLORS["muted"],
                14 if compact else 17,
                2 if compact else 3,
            )
            self._hit_regions.append(
                {
                    "rect": (x1, y1, x2, y2),
                    "action": {"type": "scenario", "value": profile.id},
                }
            )

    def render(
        self,
        camera_frame,
        detections,
        state,
        profile,
        profiles,
        motion_status,
        navigation,
        mission_map,
        lego_model_status,
        connections,
        preflight,
    ):
        canvas = np.full((self.height, self.width, 3), COLORS["background"], dtype=np.uint8)
        self._hit_regions = []
        self._draw_header(
            canvas,
            state,
            profile,
            motion_status,
            navigation,
            connections,
        )
        target_selectable = profile.navigation_policy in ("target", "lego", "rescue")
        self._draw_video(
            canvas,
            camera_frame,
            detections,
            target_selectable,
        )
        sidebar_left = self.width - self.sidebar_width
        cv2.rectangle(
            canvas,
            (sidebar_left, self.header_height),
            (self.width, self.height),
            COLORS["panel"],
            -1,
        )
        top = self._draw_controls(
            canvas,
            state,
            motion_status,
            target_selectable,
        )
        top = self._draw_tof(canvas, motion_status["ranges"], top)
        top = self._draw_objects(canvas, detections, top)
        top = self._draw_map(canvas, mission_map, motion_status["position"], top)
        self._draw_events(canvas, top)
        self._draw_captions(canvas, state)

        self._text(
            canvas,
            (
                f"YOLO {state.get('yolo_inference_ms', 0):.0f} ms | "
                f"LEGO: {lego_model_status}"
            )[:48],
            (sidebar_left + 12, self.height - 14),
            0.31,
            COLORS["muted"],
        )
        if self.show_scenario_menu:
            self._draw_scenario_menu(canvas, profiles, profile.id, preflight)
        return canvas
