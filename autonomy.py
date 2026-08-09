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


VALID_TOF_DIRECTIONS = frozenset(("front", "left", "right", "rear"))


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
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
        self.last_gimbal_command = (0.0, 0.0)

    def connect(self):
        self.connected = True

    def drive_speed(self, x, y, z, timeout=0.35):
        self.last_command = MotionCommand(x, y, z, "dry-run command")

    def stop(self):
        self.last_command = MotionCommand(reason="stopped")
        self.last_gimbal_command = (0.0, 0.0)

    def drive_gimbal(self, pitch_speed, yaw_speed):
        self.last_gimbal_command = (float(pitch_speed), float(yaw_speed))

    def stop_gimbal(self):
        self.last_gimbal_command = (0.0, 0.0)

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
        self.gimbal = None
        self._lock = threading.Lock()
        self._distances = ()
        self._distance_updated_at = 0.0
        self._position = (0.0, 0.0, 0.0)
        self._impact = False
        self._emergency_callback = None

    def set_emergency_callback(self, callback):
        self._emergency_callback = callback

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
        self.gimbal = getattr(self.robot, "gimbal", None)

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
                newly_latched = not self._impact
                self._impact = True
            if not newly_latched:
                return
            if self._emergency_callback is not None:
                self._emergency_callback("hardware impact detected")
            else:
                self.stop()

    def drive_speed(self, x, y, z, timeout=0.35):
        if not self.connected:
            raise RuntimeError("RoboMaster is not connected.")
        self.chassis.drive_speed(x=x, y=y, z=z, timeout=timeout)

    def stop(self):
        # Stop both actuators even if one SDK call fails.  In particular, a
        # disconnected chassis must never prevent a still-live gimbal from
        # receiving its zero-speed command.
        errors = []
        if self.chassis is not None:
            try:
                self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
            except Exception as error:
                errors.append(f"chassis: {error}")
        try:
            self.stop_gimbal()
        except Exception as error:
            errors.append(f"gimbal: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def drive_gimbal(self, pitch_speed, yaw_speed):
        if not self.connected or self.gimbal is None:
            raise RuntimeError("RoboMaster gimbal is unavailable.")
        self.gimbal.drive_speed(
            pitch_speed=float(pitch_speed),
            yaw_speed=float(yaw_speed),
        )

    def stop_gimbal(self):
        if self.gimbal is not None:
            self.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

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
        errors = []
        try:
            self.stop()
        except Exception as error:
            errors.append(f"stop: {error}")
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
            try:
                self.robot.close()
            except Exception as error:
                errors.append(f"robot close: {error}")
        self.connected = False
        if errors:
            raise RuntimeError("; ".join(errors))


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
        max_gimbal_dps=25.0,
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
        self.max_gimbal_dps = max_gimbal_dps
        self.sensor_timeout_s = sensor_timeout_s
        self.watchdog_timeout_s = watchdog_timeout_s
        self.logger = logger or MissionLogger()

        if not self.tof_layout:
            raise ValueError("At least one ToF direction must be configured.")
        if len(set(self.tof_layout)) != len(self.tof_layout):
            raise ValueError("ToF direction mappings must be unique.")
        unsupported = set(self.tof_layout) - VALID_TOF_DIRECTIONS
        if unsupported:
            raise ValueError(
                "Unsupported ToF direction mappings: " + ", ".join(sorted(unsupported))
            )
        if not 1 <= self.min_tof_count <= len(self.tof_layout):
            raise ValueError("min_tof_count must fit the configured ToF layout.")

        self.armed = False
        self.emergency_latched = False
        self.last_reason = "motion disabled"
        self.last_command_at = time.monotonic()
        self._closed = threading.Event()
        self._command_lock = threading.RLock()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="motion-watchdog",
            daemon=True,
        )
        if hasattr(self.backend, "set_emergency_callback"):
            self.backend.set_emergency_callback(self.emergency_stop)

    def connect(self):
        with self._command_lock:
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
        with self._command_lock:
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
        with self._command_lock:
            was_armed = self.armed
            self.armed = False
            self.last_reason = reason
            self._send_stop_locked(reason)
            if was_armed:
                self.logger.record("motion_disarmed", reason=self.last_reason)

    def stop(self, reason="stop", latch=False):
        with self._command_lock:
            already_latched = self.emergency_latched
            self.last_reason = reason
            if latch:
                # Latch controller state before attempting an SDK call that may fail.
                self.armed = False
                self.emergency_latched = True
            self._send_stop_locked(reason)
            if latch and not already_latched:
                self.logger.record("emergency_stop", reason=self.last_reason)

    def _send_stop_locked(self, reason):
        try:
            self.backend.stop()
            return True
        except Exception as error:
            already_latched = self.emergency_latched
            self.armed = False
            self.emergency_latched = True
            self.last_reason = f"{reason}; stop command failed: {error}"
            if not already_latched:
                self.logger.record("emergency_stop", reason=self.last_reason)
            return False

    def emergency_stop(self, reason="operator emergency stop"):
        self.stop(reason, latch=True)

    def reset_emergency_stop(self):
        with self._command_lock:
            if not self.emergency_latched:
                return False
            self.armed = False
            if not self._send_stop_locked("emergency reset failed"):
                return False
            try:
                self.backend.clear_impact()
            except Exception as error:
                self.emergency_latched = True
                self.last_reason = f"impact reset failed: {error}"
                return False
            self.emergency_latched = False
            self.last_reason = "emergency reset; press M to arm"
            self.logger.record("emergency_reset")
            return True

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
        with self._command_lock:
            self.last_command_at = time.monotonic()
            snapshot = self.backend.snapshot()

            if self.emergency_latched:
                return MotionCommand(reason=self.last_reason)
            if snapshot.impact:
                self.emergency_stop("impact detected")
                return MotionCommand(reason=self.last_reason)
            if not self.armed:
                return MotionCommand(reason=self.last_reason)
            if not self.sensor_ready(snapshot):
                self.emergency_stop("ToF data missing or stale")
                return MotionCommand(reason=self.last_reason)

            ranges = self._ranges(snapshot)
            forward = max(
                -self.max_forward_mps,
                min(self.max_forward_mps, command.forward_mps),
            )
            lateral = max(
                -self.max_forward_mps,
                min(self.max_forward_mps, command.lateral_mps),
            )
            yaw = max(-self.max_yaw_dps, min(self.max_yaw_dps, command.yaw_dps))

            if forward > 0:
                front_distance = ranges.get("front", 0)
                required_clearance = self.wall_clearance_mm
                if any(item.get("label") == "person" for item in detections):
                    required_clearance = max(
                        required_clearance,
                        self.person_clearance_mm,
                    )
                if front_distance < required_clearance:
                    self.stop(
                        f"forward blocked at {front_distance:.0f} mm",
                        latch=False,
                    )
                    return MotionCommand(reason=self.last_reason)
                if self._person_too_close(detections, frame_shape):
                    self.stop("visual person safety stop", latch=False)
                    return MotionCommand(reason=self.last_reason)
            elif forward < 0 and ranges.get("rear", 0) < self.wall_clearance_mm:
                rear_distance = ranges.get("rear", 0)
                self.stop(
                    f"reverse blocked at {rear_distance:.0f} mm",
                    latch=False,
                )
                return MotionCommand(reason=self.last_reason)

            side = "left" if lateral > 0 else "right"
            if lateral and ranges.get(side, 0) < self.wall_clearance_mm:
                side_distance = ranges.get(side, 0)
                self.stop(
                    f"{side} movement blocked at {side_distance:.0f} mm",
                    latch=False,
                )
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

    def apply_gimbal(self, pitch_dps, yaw_dps):
        """Send a bounded gimbal command under the asynchronous safety lock."""
        with self._command_lock:
            if self.emergency_latched or not self.armed:
                return False
            pitch = max(
                -self.max_gimbal_dps,
                min(self.max_gimbal_dps, float(pitch_dps)),
            )
            yaw = max(
                -self.max_gimbal_dps,
                min(self.max_gimbal_dps, float(yaw_dps)),
            )
            try:
                self.backend.drive_gimbal(pitch, yaw)
                return True
            except Exception as error:
                self.emergency_stop(f"gimbal command failed: {error}")
                return False

    def stop_gimbal(self):
        """Stop only the gimbal; failure latches the full emergency stop."""
        with self._command_lock:
            try:
                self.backend.stop_gimbal()
                return True
            except Exception as error:
                self.emergency_stop(f"gimbal stop failed: {error}")
                return False

    def status(self):
        with self._command_lock:
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
            with self._command_lock:
                if (
                    self.armed
                    and time.monotonic() - self.last_command_at
                    > self.watchdog_timeout_s
                ):
                    self.emergency_stop("motion watchdog timeout")

    def close(self):
        with self._command_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            self.armed = False
            self._send_stop_locked("controller closing")
            try:
                self.backend.close()
            except Exception as error:
                self.emergency_latched = True
                self.last_reason = f"backend close failed: {error}"
        if self._watchdog.is_alive():
            self._watchdog.join(timeout=1)


class AutonomousPlanner:
    """Converts mission state and detections into low-speed motion requests."""

    def __init__(
        self,
        forward_speed_mps=0.10,
        yaw_speed_dps=10.0,
        patrol_turn_distance_mm=900,
        target_stop_height_ratio=0.35,
    ):
        self.forward_speed_mps = forward_speed_mps
        self.yaw_speed_dps = yaw_speed_dps
        self.patrol_turn_distance_mm = patrol_turn_distance_mm
        self.target_stop_height_ratio = target_stop_height_ratio

    def _approach(self, detections, target_label, frame_shape):
        targets = [item for item in detections if item.get("label") == target_label]
        if not targets:
            return MotionCommand(yaw_dps=self.yaw_speed_dps, reason=f"scan for {target_label}")

        target = max(targets, key=lambda item: item.get("confidence", 0.0))
        x1, y1, x2, y2 = target["box"]
        frame_height, frame_width = frame_shape[:2]
        center_x = (x1 + x2) / 2
        height_ratio = (y2 - y1) / max(frame_height, 1)

        if height_ratio >= self.target_stop_height_ratio:
            return MotionCommand(reason=f"stop - {target_label} close")

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
