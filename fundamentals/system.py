"""Input, process, memory, output, and feedback in one visible loop.

This module deliberately has no robot SDK dependency.  It lets a class inspect
every value before mapping an output to real hardware in RoboMaster Lab.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OutputKind(str, Enum):
    """A small output vocabulary that is easy to test."""

    HOLD = "HOLD"
    STATUS = "STATUS"
    STOP_AND_ALERT = "STOP_AND_ALERT"
    OBJECT_CONFIRMED = "OBJECT_CONFIRMED"


@dataclass(frozen=True)
class RobotInput:
    """Raw values entering one software tick."""

    sequence: int
    button: str = ""
    distance_cm: float | None = None
    frame_id: str | None = None


@dataclass(frozen=True)
class ProcessedInput:
    """Values after Python has checked and named the raw input."""

    sequence: int
    button: str
    distance_cm: float | None
    distance_band: str
    frame_id: str | None


@dataclass(frozen=True)
class MemorySnapshot:
    """A read-only picture of working memory at one moment."""

    last_sequence: int
    tick_count: int
    previous_distance_cm: float | None
    last_output: str


@dataclass(frozen=True)
class RobotOutput:
    """Visible output plus an explicit chassis safety state."""

    kind: OutputKind
    chassis: str
    led: str
    screen: str
    sound: str = "NONE"


@dataclass(frozen=True)
class SystemTrace:
    """Evidence for every stage of one tick."""

    input: RobotInput
    processed: ProcessedInput
    memory_before: MemorySnapshot
    decision: str
    output: RobotOutput
    memory_after: MemorySnapshot
    feedback: str


class WorkingMemory:
    """Short-lived state that is reset whenever the program restarts."""

    def __init__(self) -> None:
        self.last_sequence = -1
        self.tick_count = 0
        self.previous_distance_cm: float | None = None
        self.last_output = OutputKind.HOLD.value

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            last_sequence=self.last_sequence,
            tick_count=self.tick_count,
            previous_distance_cm=self.previous_distance_cm,
            last_output=self.last_output,
        )

    def remember(self, processed: ProcessedInput, output: RobotOutput) -> None:
        self.last_sequence = processed.sequence
        self.tick_count += 1
        self.previous_distance_cm = processed.distance_cm
        self.last_output = output.kind.value


def process_input(raw: RobotInput) -> ProcessedInput:
    """Validate raw values and turn numbers into useful categories."""

    if raw.sequence < 0:
        raise ValueError("sequence must be zero or greater")
    if raw.distance_cm is not None and raw.distance_cm < 0:
        raise ValueError("distance_cm cannot be negative")

    button = raw.button.strip().upper()
    if raw.distance_cm is None:
        distance_band = "UNKNOWN"
    elif raw.distance_cm < 50:
        distance_band = "NEAR"
    else:
        distance_band = "CLEAR"

    return ProcessedInput(
        sequence=raw.sequence,
        button=button,
        distance_cm=raw.distance_cm,
        distance_band=distance_band,
        frame_id=raw.frame_id,
    )


def decide(processed: ProcessedInput, memory: MemorySnapshot) -> str:
    """Use transparent rules; this function is automation, not learned AI."""

    if processed.sequence <= memory.last_sequence:
        return "STALE_INPUT"
    if processed.button in {"STOP", "E-STOP", "ESTOP"}:
        return "HUMAN_STOP"
    if processed.distance_band == "NEAR":
        return "NEAR_OBJECT"
    if processed.distance_band == "UNKNOWN":
        return "WAIT_FOR_INPUT"
    return "READY"


def make_output(decision: str) -> RobotOutput:
    """Translate a decision into observable outputs.

    The laptop project never emits physical motion.  Chassis remains STOP until
    a separately reviewed RoboMaster Lab program and a human allow it.
    """

    if decision in {"HUMAN_STOP", "NEAR_OBJECT"}:
        return RobotOutput(
            kind=OutputKind.STOP_AND_ALERT,
            chassis="STOP",
            led="RED",
            screen=decision,
            sound="ALERT",
        )
    if decision in {"STALE_INPUT", "WAIT_FOR_INPUT"}:
        return RobotOutput(
            kind=OutputKind.HOLD,
            chassis="STOP",
            led="AMBER",
            screen=decision,
        )
    return RobotOutput(
        kind=OutputKind.STATUS,
        chassis="STOP",
        led="BLUE",
        screen="READY: observation only",
    )


class RobotSystem:
    """Run the same six stages once at a time."""

    def __init__(self) -> None:
        self.memory = WorkingMemory()

    def tick(self, raw: RobotInput) -> SystemTrace:
        before = self.memory.snapshot()
        processed = process_input(raw)
        decision = decide(processed, before)
        output = make_output(decision)

        if decision != "STALE_INPUT":
            self.memory.remember(processed, output)
        after = self.memory.snapshot()
        feedback = (
            f"next tick will read after sequence {after.last_sequence}; "
            f"last output was {after.last_output}"
        )
        return SystemTrace(
            input=raw,
            processed=processed,
            memory_before=before,
            decision=decision,
            output=output,
            memory_after=after,
            feedback=feedback,
        )
