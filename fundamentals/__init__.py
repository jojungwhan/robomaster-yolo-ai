"""Small, visible building blocks for the RoboMaster systems course."""

from .agent import AgentTrace, RobotAgent
from .communication import (
    LockedMessageError,
    Message,
    SecretBox,
    WirelessLink,
    decode_message,
    encode_message,
)
from .system import (
    MemorySnapshot,
    OutputKind,
    RobotInput,
    RobotOutput,
    RobotSystem,
    SystemTrace,
)
from .vision import (
    AIPrediction,
    Frame,
    ObservationMemory,
    RiskDecision,
    RiskPolicy,
    measure_bright_region,
)

__all__ = [
    "AIPrediction",
    "AgentTrace",
    "Frame",
    "LockedMessageError",
    "MemorySnapshot",
    "Message",
    "ObservationMemory",
    "OutputKind",
    "RiskDecision",
    "RiskPolicy",
    "RobotAgent",
    "RobotInput",
    "RobotOutput",
    "RobotSystem",
    "SecretBox",
    "SystemTrace",
    "WirelessLink",
    "decode_message",
    "encode_message",
    "measure_bright_region",
]
