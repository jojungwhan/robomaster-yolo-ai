"""Safety-gated motion control for the RoboMaster vision demo.

Physical motion is deliberately two-stage: the process must be started with motion
enabled and the operator must then arm it. Forward/turn commands require fresh ToF
readings, and every hardware command has a short SDK timeout.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time


@dataclass(frozen=True)
class MotionCommand:
    forward_mps: float = 0.0
    lateral_mps: float = 0.0
    yaw_dps: float = 0.0
    reason: str = "stop"


@dataclass(frozen=True)
class SensorSnapshot:
    distances_mm: tuple = ()
    distance_age_s: float = float("inf")
    position: tuple = (0.0, 0.0, 0.0)
    impact: bool = False


class MissionLogger:
    def __init__(self, path="mission_events.jsonl"):
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, event, **data):
        payload = {
            "timestamp": time.time(),
            "event": event,
            **data,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


class DryRunBackend:
    """Simulates a four-ToF robot and records commands without moving hardware."""

    name = "dry-run"

    def __init__(self, simulated_distance_mm=2000):
        self.connected = False
        self.last_command = MotionCommand()
        self.distances = (simulated_distance_mm,) * 4
        self.distance_updated_at = time.monotonic()
        self.position = (0.0, 0.0, 0.0)
        self.impact = False

    def connect(self):
        self.connected = True

    def drive_speed(self, x, y, z, timeout=0.35):
        self.last_command = MotionCommand(x, y, z, "dry-run command")

    def stop(self):
        self.last_command = MotionCommand(reason="stopped")

    def snapshot(self):
        return SensorSnapshot(
            distances_mm=tuple(self.distances),
            distance_age_s=0.0,
            position=tuple(self.position),
            impact=self.impact,
        )

    def set_distances(self, distances_mm):
        self.distances = tuple(distances_mm)
        self.distance_updated_at = time.monotonic()

    def clear_impact(self):
        self.impact = False

    def close(self):
        self.stop()
        self.connected = False


class RoboMasterBackend:
    """Optional adapter for the official/community RoboMaster Python client."""

    name = "robomaster-sdk"

    def __init__(self, conn_type="ap"):
        self.conn_type = conn_type
        self.connected = False
        self.robot = None
        self.chassis = None
        self.distance_sensor = None
        self._lock = threading.Lock()
        self._distances = ()
        self._distance_updated_at = 0.0
        self._position = (0.0, 0.0, 0.0)
        self._impact = False

    def connect(self):
        try:
            from robomaster import robot
        except ImportError as error:
            raise RuntimeError(
                "RoboMaster SDK is not installed. Use the official EP SDK or a "
                "compatible S1 driver before enabling hardware motion."
            ) from error

        self.robot = robot.Robot()
        self.robot.initialize(conn_type=self.conn_type)
        self.chassis = self.robot.chassis
        self.distance_sensor = self.robot.sensor

        if not self.chassis.sub_position(
            cs=1, freq=10, callback=self._on_position
        ):
            raise RuntimeError("Could not subscribe to RoboMaster odometry.")
        if not self.chassis.sub_status(freq=10, callback=self._on_status):
            raise RuntimeError("Could not subscribe to RoboMaster chassis status.")
        if not self.distance_sensor.sub_distance(
            freq=20, callback=self._on_distance
        ):
            raise RuntimeError(
                "Could not subscribe to ToF sensors. Hardware motion remains disabled."
            )

        self.connected = True

    def _on_distance(self, distances):
        values = tuple(float(value) for value in distances)
        with self._lock:
            self._distances = values
            self._distance_updated_at = time.monotonic()

    def _on_position(self, position):
        with self._lock:
            self._position = tuple(float(value) for value in position[:3])

    def _on_status(self, status):
        # SDK status order: static, slopes, pickup, slip, impact x/y/z, rollover.
        impact = len(status) >= 9 and any(bool(value) for value in status[6:9])
        if impact:
            with self._lock:
                self._impact = True
            self.stop()

    def drive_speed(self, x, y, z, timeout=0.35):
        if not self.connected:
            raise RuntimeError("RoboMaster is not connected.")
        self.chassis.drive_speed(x=x, y=y, z=z, timeout=timeout)

    def stop(self):
        if self.chassis is not None:
            self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)

    def snapshot(self):
        with self._lock:
            age = (
                time.monotonic() - self._distance_updated_at
                if self._distance_updated_at
                else float("inf")
            )
            return SensorSnapshot(
                distances_mm=self._distances,
                distance_age_s=age,
                position=self._position,
                impact=self._impact,
            )

    def clear_impact(self):
        with self._lock:
            self._impact = False

    def close(self):
        self.stop()
        if self.chassis is not None:
            for unsubscribe in (
                self.chassis.unsub_position,
                self.chassis.unsub_status,
            ):
                try:
                    unsubscribe()
                except Exception:
                    pass
        if self.distance_sensor is not None:
            try:
                self.distance_sensor.unsub_distance()
            except Exception:
                pass
        if self.robot is not None:
            self.robot.close()
        self.connected = False


class SafetyMotionController:
    """Applies speed, range, vision, impact, and watchdog interlocks."""

    def __init__(
        self,
        backend,
        motion_requested=False,
        tof_layout=("front", "left", "right", "rear"),
        min_tof_count=4,
        wall_clearance_mm=700,
        person_clearance_mm=1200,
        rotation_clearance_mm=450,
        max_forward_mps=0.12,
        max_yaw_dps=12.0,
        sensor_timeout_s=0.6,
        watchdog_timeout_s=0.7,
        logger=None,
    ):
        self.backend = backend
        self.motion_requested = motion_requested
        self.tof_layout = tuple(tof_layout)
        self.min_tof_count = min_tof_count
        self.wall_clearance_mm = wall_clearance_mm
        self.person_clearance_mm = person_clearance_mm
        self.rotation_clearance_mm = rotation_clearance_mm
        self.max_forward_mps = max_forward_mps
        self.max_yaw_dps = max_yaw_dps
        self.sensor_timeout_s = sensor_timeout_s
        self.watchdog_timeout_s = watchdog_timeout_s
        self.logger = logger or MissionLogger()

        self.armed = False
        self.emergency_latched = False
        self.last_reason = "motion disabled"
        self.last_command_at = time.monotonic()
        self._closed = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="motion-watchdog",
            daemon=True,
        )

    def connect(self):
        self.backend.connect()
        self.last_reason = f"{self.backend.name} connected; press M to arm"
        self.logger.record("motion_backend_connected", backend=self.backend.name)
        self._watchdog.start()

    def _ranges(self, snapshot=None):
        snapshot = snapshot or self.backend.snapshot()
        return {
            direction: snapshot.distances_mm[index]
            for index, direction in enumerate(self.tof_layout)
            if index < len(snapshot.distances_mm)
        }

    def sensor_ready(self, snapshot=None):
        snapshot = snapshot or self.backend.snapshot()
        return (
            snapshot.distance_age_s <= self.sensor_timeout_s
            and len(snapshot.distances_mm) >= self.min_tof_count
            and all(value > 0 for value in snapshot.distances_mm)
        )

    def arm(self):
        snapshot = self.backend.snapshot()
        if not self.motion_requested:
            self.last_reason = "restart with --enable-motion before arming"
            return False
        if self.emergency_latched:
            self.last_reason = "emergency stop latched; press R to reset"
            return False
        if not self.sensor_ready(snapshot):
            self.stop("ToF safety check failed", latch=False)
            return False

        self.armed = True
        self.last_command_at = time.monotonic()
        self.last_reason = "armed"
        self.logger.record(
            "motion_armed",
            backend=self.backend.name,
            distances_mm=list(snapshot.distances_mm),
        )
        return True

    def disarm(self, reason="operator disarmed"):
        self.armed = False
        self.backend.stop()
        self.last_reason = reason
        self.logger.record("motion_disarmed", reason=reason)

    def stop(self, reason="stop", latch=False):
        self.backend.stop()
        self.last_reason = reason
        if latch:
            self.armed = False
            self.emergency_latched = True
            self.logger.record("emergency_stop", reason=reason)

    def emergency_stop(self, reason="operator emergency stop"):
        self.stop(reason, latch=True)

    def reset_emergency_stop(self):
        self.backend.stop()
        self.backend.clear_impact()
        self.armed = False
        self.emergency_latched = False
        self.last_reason = "emergency reset; press M to arm"
        self.logger.record("emergency_reset")

    @staticmethod
    def _person_too_close(detections, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        for detection in detections:
            if detection.get("label") != "person":
                continue
            x1, y1, x2, y2 = detection["box"]
            height_ratio = (y2 - y1) / max(frame_height, 1)
            center_x = (x1 + x2) / 2
            in_forward_corridor = frame_width * 0.25 <= center_x <= frame_width * 0.75
            if height_ratio >= 0.34 and in_forward_corridor:
                return True
        return False

    def apply(self, command, detections, frame_shape):
        self.last_command_at = time.monotonic()
        snapshot = self.backend.snapshot()

        if snapshot.impact:
            self.emergency_stop("impact detected")
            return MotionCommand(reason=self.last_reason)
        if not self.armed:
            self.backend.stop()
            return MotionCommand(reason=self.last_reason)
        if not self.sensor_ready(snapshot):
            self.emergency_stop("ToF data missing or stale")
            return MotionCommand(reason=self.last_reason)

        ranges = self._ranges(snapshot)
        forward = max(-self.max_forward_mps, min(self.max_forward_mps, command.forward_mps))
        lateral = max(-self.max_forward_mps, min(self.max_forward_mps, command.lateral_mps))
        yaw = max(-self.max_yaw_dps, min(self.max_yaw_dps, command.yaw_dps))

        if forward > 0:
            front_distance = ranges.get("front", 0)
            required_clearance = self.wall_clearance_mm
            if any(item.get("label") == "person" for item in detections):
                required_clearance = max(required_clearance, self.person_clearance_mm)
            if front_distance < required_clearance:
                self.stop(
                    f"forward blocked at {front_distance:.0f} mm",
                    latch=False,
                )
                return MotionCommand(reason=self.last_reason)
            if self._person_too_close(detections, frame_shape):
                self.stop("visual person safety stop", latch=False)
                return MotionCommand(reason=self.last_reason)

        if yaw and min(ranges.values(), default=0) < self.rotation_clearance_mm:
            nearest = min(ranges.values(), default=0)
            self.stop(f"rotation blocked at {nearest:.0f} mm", latch=False)
            return MotionCommand(reason=self.last_reason)

        safe_command = MotionCommand(forward, lateral, yaw, command.reason)
        try:
            self.backend.drive_speed(
                x=safe_command.forward_mps,
                y=safe_command.lateral_mps,
                z=safe_command.yaw_dps,
                timeout=0.35,
            )
            self.last_reason = safe_command.reason
            return safe_command
        except Exception as error:
            self.emergency_stop(f"motion command failed: {error}")
            return MotionCommand(reason=self.last_reason)

    def status(self):
        snapshot = self.backend.snapshot()
        if self.emergency_latched:
            mode = "ESTOP"
        elif self.armed:
            mode = "ARMED"
        else:
            mode = "DISARMED"
        return {
            "mode": mode,
            "backend": self.backend.name,
            "reason": self.last_reason,
            "ranges": self._ranges(snapshot),
            "position": snapshot.position,
            "sensor_ready": self.sensor_ready(snapshot),
        }

    def _watchdog_loop(self):
        while not self._closed.wait(0.1):
            if self.armed and time.monotonic() - self.last_command_at > self.watchdog_timeout_s:
                self.emergency_stop("motion watchdog timeout")

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        self.armed = False
        self.backend.close()
        if self._watchdog.is_alive():
            self._watchdog.join(timeout=1)


class AutonomousPlanner:
    """Converts mission state and detections into low-speed motion requests."""

    def __init__(
        self,
        forward_speed_mps=0.10,
        yaw_speed_dps=10.0,
        patrol_turn_distance_mm=900,
    ):
        self.forward_speed_mps = forward_speed_mps
        self.yaw_speed_dps = yaw_speed_dps
        self.patrol_turn_distance_mm = patrol_turn_distance_mm

    def _approach(self, detections, target_label, frame_shape):
        targets = [item for item in detections if item.get("label") == target_label]
        if not targets:
            return MotionCommand(yaw_dps=self.yaw_speed_dps, reason=f"scan for {target_label}")

        target = max(targets, key=lambda item: item.get("confidence", 0.0))
        x1, _y1, x2, _y2 = target["box"]
        frame_width = frame_shape[1]
        center_x = (x1 + x2) / 2

        if center_x < frame_width * 0.43:
            return MotionCommand(yaw_dps=self.yaw_speed_dps, reason=f"align left to {target_label}")
        if center_x > frame_width * 0.57:
            return MotionCommand(yaw_dps=-self.yaw_speed_dps, reason=f"align right to {target_label}")
        return MotionCommand(forward_mps=self.forward_speed_mps, reason=f"approach {target_label}")

    def _patrol(self, ranges):
        front = ranges.get("front", 0)
        left = ranges.get("left", 0)
        right = ranges.get("right", 0)

        if front < self.patrol_turn_distance_mm:
            if left >= right:
                return MotionCommand(yaw_dps=self.yaw_speed_dps, reason="avoid obstacle left")
            return MotionCommand(yaw_dps=-self.yaw_speed_dps, reason="avoid obstacle right")

        return MotionCommand(forward_mps=self.forward_speed_mps, reason="slow patrol")

    def plan(self, scenario_id, detections, target_label, frame_shape, ranges):
        if scenario_id == "rescue":
            if any(item.get("label") == "person" for item in detections):
                return self._approach(detections, "person", frame_shape)
            return self._patrol(ranges)
        if scenario_id in ("target", "lego"):
            return self._approach(detections, target_label, frame_shape)
        return self._patrol(ranges)
