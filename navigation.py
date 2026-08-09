"""Mission state, persistent target tracking, waypoints, and odometry mapping."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
import time

from autonomy import AutonomousPlanner, MotionCommand


class NavigationState(str, Enum):
    DISARMED = "DISARMED"
    READY = "READY"
    PAUSED = "PAUSED"
    SCAN = "SCAN"
    ACQUIRING = "ACQUIRING TARGET"
    PATROL = "PATROL"
    TARGET_LOCKED = "TARGET LOCKED"
    APPROACH = "APPROACH"
    INTERACT = "STOP / INTERACT"
    RECOVERY = "RECOVERY"
    WAYPOINT = "WAYPOINT"
    COMPLETE = "COMPLETE"
    ESTOP = "ESTOP"


@dataclass(frozen=True)
class TargetObservation:
    detection: dict | None
    locked: bool
    just_locked: bool
    temporarily_lost: bool
    age_seconds: float


def box_iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / max(union, 1)


class TargetTracker:
    """Lock one label/instance across transient detector misses and nearby distractors."""

    def __init__(
        self,
        acquire_frames=2,
        lost_timeout_seconds=1.0,
        reacquire_timeout_seconds=3.0,
        minimum_iou=0.08,
    ):
        self.acquire_frames = acquire_frames
        self.lost_timeout_seconds = lost_timeout_seconds
        self.reacquire_timeout_seconds = reacquire_timeout_seconds
        self.minimum_iou = minimum_iou
        self.reset()

    def reset(self, label=None):
        self.label = label
        self.last_box = None
        self.last_detection = None
        self.last_seen_at = float("-inf")
        self.hits = 0
        self.locked = False
        self.last_observation_id = None

    def seed(self, detection, now=None):
        now = time.monotonic() if now is None else now
        self.reset(detection.get("label"))
        self.last_box = tuple(detection["box"])
        self.last_detection = dict(detection)
        self.last_seen_at = now
        self.hits = 1
        self.locked = self.hits >= self.acquire_frames
        self.last_observation_id = detection.get("observation_id")

    def update(self, detections, label, now=None):
        now = time.monotonic() if now is None else now
        if label != self.label:
            self.reset(label)

        candidates = [item for item in detections if item.get("label") == label]
        selected = None
        if candidates and self.last_box is None:
            selected = max(candidates, key=lambda item: item.get("confidence", 0.0))
        elif candidates:
            ranked = sorted(
                candidates,
                key=lambda item: (
                    box_iou(self.last_box, item["box"]),
                    item.get("confidence", 0.0),
                ),
                reverse=True,
            )
            best_iou = box_iou(self.last_box, ranked[0]["box"])
            if best_iou >= self.minimum_iou:
                selected = ranked[0]
            elif now - self.last_seen_at > self.reacquire_timeout_seconds:
                self.reset(label)
                selected = max(
                    candidates,
                    key=lambda item: item.get("confidence", 0.0),
                )

        if selected is not None:
            was_locked = self.locked
            self.last_box = tuple(selected["box"])
            self.last_detection = dict(selected)
            self.last_seen_at = now
            observation_id = selected.get("observation_id")
            is_new_observation = (
                observation_id is None
                or observation_id != self.last_observation_id
            )
            if is_new_observation:
                self.hits += 1
            self.last_observation_id = observation_id
            self.locked = self.hits >= self.acquire_frames
            return TargetObservation(
                dict(selected),
                self.locked,
                self.locked and not was_locked,
                False,
                0.0,
            )

        age = now - self.last_seen_at
        temporarily_lost = self.last_box is not None and age <= self.lost_timeout_seconds
        if age > self.reacquire_timeout_seconds:
            self.reset(label)
            age = float("inf")
        return TargetObservation(None, self.locked, False, temporarily_lost, age)


@dataclass
class Waypoint:
    name: str
    x: float
    y: float


class MissionMap:
    """Record a lightweight odometry trail, range endpoints, and named waypoints."""

    DIRECTION_OFFSETS_DEGREES = {
        "front": 0,
        "left": 90,
        "right": -90,
        "rear": 180,
    }

    def __init__(self, max_trail_points=2000, max_obstacle_points=1000):
        self.trail = deque(maxlen=max_trail_points)
        self.obstacles = deque(maxlen=max_obstacle_points)
        self.waypoints = []
        self.home = None
        self.active_waypoint_index = None

    @staticmethod
    def _xy_heading(position):
        values = tuple(float(value) for value in position)
        x = values[0] if values else 0.0
        y = values[1] if len(values) > 1 else 0.0
        heading = values[2] if len(values) > 2 else 0.0
        return x, y, heading

    def record(self, position, ranges):
        x, y, heading = self._xy_heading(position)
        if self.home is None:
            self.home = (x, y)
        if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) >= 0.02:
            self.trail.append((x, y))

        for direction, distance_mm in ranges.items():
            if direction not in self.DIRECTION_OFFSETS_DEGREES:
                continue
            if not 80 <= distance_mm <= 4000:
                continue
            angle = math.radians(
                heading + self.DIRECTION_OFFSETS_DEGREES[direction]
            )
            distance_m = distance_mm / 1000.0
            self.obstacles.append(
                (
                    x + math.cos(angle) * distance_m,
                    y + math.sin(angle) * distance_m,
                )
            )

    def add_waypoint(self, name, position):
        x, y, _heading = self._xy_heading(position)
        waypoint = Waypoint(name=name, x=x, y=y)
        self.waypoints.append(waypoint)
        self.active_waypoint_index = len(self.waypoints) - 1
        return waypoint

    def select_waypoint(self, index):
        if index is None:
            self.active_waypoint_index = None
            return None
        if not 0 <= index < len(self.waypoints):
            raise IndexError("Waypoint index is out of range.")
        self.active_waypoint_index = index
        return self.waypoints[index]

    def active_goal(self, return_home=False):
        if return_home:
            return self.home
        if self.active_waypoint_index is None:
            return None
        waypoint = self.waypoints[self.active_waypoint_index]
        return waypoint.x, waypoint.y

    def save(self, path):
        payload = {
            "version": 1,
            "home": list(self.home) if self.home is not None else None,
            "trail": [list(point) for point in self.trail],
            "obstacles": [list(point) for point in self.obstacles],
            "waypoints": [waypoint.__dict__ for waypoint in self.waypoints],
            "active_waypoint_index": self.active_waypoint_index,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        mission_map = cls()
        path = Path(path)
        if not path.is_file():
            return mission_map
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("Unsupported mission map version.")
        mission_map.home = tuple(payload["home"]) if payload.get("home") else None
        mission_map.trail.extend(tuple(point) for point in payload.get("trail", ()))
        mission_map.obstacles.extend(
            tuple(point) for point in payload.get("obstacles", ())
        )
        mission_map.waypoints = [
            Waypoint(**waypoint) for waypoint in payload.get("waypoints", ())
        ]
        active = payload.get("active_waypoint_index")
        if active is not None and 0 <= active < len(mission_map.waypoints):
            mission_map.active_waypoint_index = active
        return mission_map


@dataclass(frozen=True)
class NavigationDecision:
    state: NavigationState
    command: MotionCommand
    selected_detection: dict | None
    reason: str
    distance_to_goal_m: float | None = None


@dataclass(frozen=True)
class GimbalCommand:
    pitch_dps: float = 0.0
    yaw_dps: float = 0.0
    reason: str = "gimbal hold"


class GimbalTracker:
    """Track a locked target with the camera separately from chassis navigation."""

    def __init__(self, max_speed_dps=25.0, gain=0.65, deadband_ratio=0.06):
        self.max_speed_dps = max_speed_dps
        self.gain = gain
        self.deadband_ratio = deadband_ratio

    def plan(self, detection, frame_shape):
        if detection is None:
            return GimbalCommand()
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = detection["box"]
        target_x = (x1 + x2) / 2
        target_y = (y1 + y2) / 2
        horizontal_error = (frame_width / 2 - target_x) / max(frame_width / 2, 1)
        vertical_error = (frame_height / 2 - target_y) / max(frame_height / 2, 1)
        if abs(horizontal_error) < self.deadband_ratio:
            horizontal_error = 0.0
        if abs(vertical_error) < self.deadband_ratio:
            vertical_error = 0.0
        yaw = max(
            -self.max_speed_dps,
            min(self.max_speed_dps, horizontal_error * self.max_speed_dps * self.gain),
        )
        pitch = max(
            -self.max_speed_dps,
            min(self.max_speed_dps, vertical_error * self.max_speed_dps * self.gain),
        )
        reason = "track locked target" if yaw or pitch else "target centered"
        return GimbalCommand(pitch, yaw, reason)


class MissionNavigator:
    """Explicit mission state machine layered above the safety controller."""

    def __init__(
        self,
        planner=None,
        tracker=None,
        mission_map=None,
        waypoint_tolerance_m=0.25,
        scan_timeout_seconds=20.0,
        scan_sweep_seconds=4.0,
    ):
        self.planner = planner or AutonomousPlanner()
        self.tracker = tracker or TargetTracker()
        self.map = mission_map or MissionMap()
        self.waypoint_tolerance_m = waypoint_tolerance_m
        self.scan_timeout_seconds = scan_timeout_seconds
        self.scan_sweep_seconds = scan_sweep_seconds
        self.state = NavigationState.DISARMED
        self.reason = "motion disarmed"
        self.scenario_id = None
        self._scan_started_at = None

    def select_target(self, label, detection=None, now=None):
        if detection is not None and detection.get("label") == label:
            self.tracker.seed(detection, now=now)
        else:
            self.tracker.reset(label)

    def _decision(self, state, command, detection=None, distance=None):
        self.state = state
        self.reason = command.reason
        return NavigationDecision(state, command, detection, command.reason, distance)

    def _goal_command(self, position, goal, yaw_speed, forward_speed):
        x, y, heading = self.map._xy_heading(position)
        goal_x, goal_y = goal
        delta_x = goal_x - x
        delta_y = goal_y - y
        distance = math.hypot(delta_x, delta_y)
        if distance <= self.waypoint_tolerance_m:
            return MotionCommand(reason="goal reached"), distance

        desired_heading = math.degrees(math.atan2(delta_y, delta_x))
        heading_error = (desired_heading - heading + 180) % 360 - 180
        if abs(heading_error) > 12:
            yaw = yaw_speed if heading_error > 0 else -yaw_speed
            return MotionCommand(yaw_dps=yaw, reason="align to waypoint"), distance
        return MotionCommand(forward_mps=forward_speed, reason="approach waypoint"), distance

    def plan(
        self,
        profile,
        detections,
        target_label,
        frame_shape,
        motion_status,
        paused=False,
        interaction_active=False,
        now=None,
    ):
        now = time.monotonic() if now is None else now
        if profile.id != self.scenario_id:
            self.scenario_id = profile.id
            self.tracker.reset(target_label)
            self._scan_started_at = None

        self.map.record(motion_status["position"], motion_status["ranges"])
        observation = self.tracker.update(detections, target_label, now=now)

        if motion_status["mode"] == "ESTOP":
            return self._decision(
                NavigationState.ESTOP,
                MotionCommand(reason="emergency stop latched"),
                observation.detection,
            )
        if paused:
            return self._decision(
                NavigationState.PAUSED,
                MotionCommand(reason="mission paused"),
                observation.detection,
            )
        if interaction_active:
            return self._decision(
                NavigationState.INTERACT,
                MotionCommand(reason="interaction in progress"),
                observation.detection,
            )
        if motion_status["mode"] != "ARMED":
            return self._decision(
                NavigationState.DISARMED,
                MotionCommand(reason="motion disarmed"),
                observation.detection,
            )
        if not profile.allow_motion or profile.navigation_policy == "stationary":
            return self._decision(
                NavigationState.READY,
                MotionCommand(reason="scenario is observation only"),
                observation.detection,
            )

        if profile.navigation_policy in ("waypoint", "return_home"):
            return_home = profile.navigation_policy == "return_home"
            goal = self.map.active_goal(return_home=return_home)
            if goal is None:
                return self._decision(
                    NavigationState.READY,
                    MotionCommand(reason="select or record a waypoint"),
                )
            if (
                motion_status["ranges"].get("front", 0)
                < self.planner.patrol_turn_distance_mm
            ):
                return self._decision(
                    NavigationState.RECOVERY,
                    self.planner._patrol(motion_status["ranges"]),
                )
            command, distance = self._goal_command(
                motion_status["position"],
                goal,
                self.planner.yaw_speed_dps,
                self.planner.forward_speed_mps,
            )
            goal_completion = (
                "home_reached" if return_home else "waypoint_reached"
            )
            if distance > self.waypoint_tolerance_m:
                state = NavigationState.WAYPOINT
            elif profile.completion == goal_completion:
                state = NavigationState.COMPLETE
            else:
                state = NavigationState.INTERACT
            return self._decision(state, command, distance=distance)

        if profile.navigation_policy in ("target", "lego"):
            if not target_label:
                return self._decision(
                    NavigationState.READY,
                    MotionCommand(reason="select a target"),
                )
            if observation.detection is not None and not observation.locked:
                self._scan_started_at = None
                return self._decision(
                    NavigationState.ACQUIRING,
                    MotionCommand(reason=f"confirming {target_label} lock"),
                    observation.detection,
                )
            if observation.detection is not None and observation.locked:
                self._scan_started_at = None
                if observation.just_locked:
                    return self._decision(
                        NavigationState.TARGET_LOCKED,
                        MotionCommand(reason=f"{target_label} lock confirmed"),
                        observation.detection,
                    )
                command = self.planner._approach(
                    [observation.detection],
                    target_label,
                    frame_shape,
                )
                state = (
                    NavigationState.INTERACT
                    if "close" in command.reason
                    else NavigationState.APPROACH
                )
                return self._decision(state, command, observation.detection)
            if observation.temporarily_lost:
                return self._decision(
                    NavigationState.RECOVERY,
                    MotionCommand(reason=f"hold position; {target_label} briefly lost"),
                )
            if self._scan_started_at is None:
                self._scan_started_at = now
            scan_elapsed = now - self._scan_started_at
            if scan_elapsed >= self.scan_timeout_seconds:
                return self._decision(
                    NavigationState.RECOVERY,
                    MotionCommand(reason=f"scan timeout; {target_label} not found"),
                )
            sweep_index = int(scan_elapsed / max(self.scan_sweep_seconds, 0.1))
            scan_yaw = self.planner.yaw_speed_dps * (1 if sweep_index % 2 == 0 else -1)
            return self._decision(
                NavigationState.SCAN,
                MotionCommand(
                    yaw_dps=scan_yaw,
                    reason=f"scan for {target_label}",
                ),
            )

        command = self.planner._patrol(motion_status["ranges"])
        state = (
            NavigationState.RECOVERY
            if "avoid obstacle" in command.reason
            else NavigationState.PATROL
        )
        return self._decision(state, command)
