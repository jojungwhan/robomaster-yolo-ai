"""Computer vision, AI predictions, memory, and risk-aware thresholds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Frame:
    """A tiny grayscale image represented by numbers from 0 to 255."""

    frame_id: str
    pixels: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class VisionMeasurement:
    """A deterministic computer-vision measurement."""

    frame_id: str
    bright_fraction: float
    bright_pixel_count: int
    total_pixel_count: int


@dataclass(frozen=True)
class AIPrediction:
    """A model's uncertain observation, not a fact or a movement command."""

    frame_id: str
    label: str
    confidence: float


@dataclass(frozen=True)
class RiskDecision:
    """A Python policy decision made from an AI prediction."""

    name: str
    reason: str
    allows_motion: bool = False


@dataclass(frozen=True)
class RiskPolicy:
    """Classroom comparison values; real thresholds require validation data."""

    person_stop_threshold: float = 0.30
    object_confirm_threshold: float = 0.70
    object_fresh_frames: int = 3

    def __post_init__(self) -> None:
        for value in (self.person_stop_threshold, self.object_confirm_threshold):
            if not 0.0 <= value <= 1.0:
                raise ValueError("thresholds must be between 0 and 1")
        if self.object_fresh_frames < 1:
            raise ValueError("object_fresh_frames must be positive")


class ObservationMemory:
    """Remember consecutive predictions without counting one frame twice."""

    def __init__(self) -> None:
        self.last_frame_id: str | None = None
        self.last_label: str | None = None
        self.fresh_streak = 0

    def observe(self, prediction: AIPrediction) -> tuple[int, bool]:
        if prediction.frame_id == self.last_frame_id:
            return self.fresh_streak, False
        if prediction.label == self.last_label:
            self.fresh_streak += 1
        else:
            self.last_label = prediction.label
            self.fresh_streak = 1
        self.last_frame_id = prediction.frame_id
        return self.fresh_streak, True


def measure_bright_region(frame: Frame, pixel_threshold: int = 200) -> VisionMeasurement:
    """Measure bright pixels with a written rule: computer vision without AI."""

    if not 0 <= pixel_threshold <= 255:
        raise ValueError("pixel_threshold must be between 0 and 255")
    if not frame.pixels or not frame.pixels[0]:
        raise ValueError("frame needs at least one pixel")
    width = len(frame.pixels[0])
    if any(len(row) != width for row in frame.pixels):
        raise ValueError("all pixel rows must have the same width")

    values = [value for row in frame.pixels for value in row]
    if any(not 0 <= value <= 255 for value in values):
        raise ValueError("pixels must be between 0 and 255")
    bright = sum(value >= pixel_threshold for value in values)
    return VisionMeasurement(
        frame_id=frame.frame_id,
        bright_fraction=bright / len(values),
        bright_pixel_count=bright,
        total_pixel_count=len(values),
    )


def evaluate_prediction(
    prediction: AIPrediction,
    fresh_frames: int,
    policy: RiskPolicy | None = None,
) -> RiskDecision:
    """Choose a safe output using the class, confidence, and fresh evidence."""

    policy = policy or RiskPolicy()
    if not 0.0 <= prediction.confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    if (
        prediction.label.casefold() == "person"
        and prediction.confidence >= policy.person_stop_threshold
    ):
        return RiskDecision(
            "STOP_AND_ALERT",
            "possible person: a lower threshold may trigger protection only",
        )
    if (
        prediction.confidence >= policy.object_confirm_threshold
        and fresh_frames >= policy.object_fresh_frames
    ):
        return RiskDecision(
            "CONFIRM_OBJECT_FOR_TEACHER",
            "ordinary object repeated in enough fresh frames",
        )
    return RiskDecision(
        "HOLD_AND_OBSERVE",
        "evidence is incomplete or repeated from too few fresh frames",
    )
