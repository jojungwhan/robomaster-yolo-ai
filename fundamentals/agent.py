"""A tiny AI-agent loop built from the other fundamentals modules."""

from __future__ import annotations

from dataclasses import dataclass

from .system import OutputKind, RobotInput, RobotOutput, RobotSystem, SystemTrace
from .vision import (
    AIPrediction,
    ObservationMemory,
    RiskDecision,
    RiskPolicy,
    evaluate_prediction,
)


@dataclass(frozen=True)
class AgentTrace:
    """One observable cycle: input, state, AI observation, decision, output."""

    system: SystemTrace
    prediction: AIPrediction | None
    fresh_prediction: bool
    prediction_streak: int
    risk_decision: RiskDecision
    output: RobotOutput
    feedback: str


class RobotAgent:
    """Combine deterministic Python rules with optional AI observations."""

    def __init__(self, policy: RiskPolicy | None = None) -> None:
        self.system = RobotSystem()
        self.observations = ObservationMemory()
        self.policy = policy or RiskPolicy()

    def tick(
        self,
        raw: RobotInput,
        prediction: AIPrediction | None = None,
    ) -> AgentTrace:
        system_trace = self.system.tick(raw)

        if prediction is None:
            streak, fresh = 0, False
            risk = RiskDecision("HOLD_AND_OBSERVE", "no AI observation this tick")
        else:
            streak, fresh = self.observations.observe(prediction)
            risk = evaluate_prediction(prediction, streak, self.policy)

        if system_trace.output.kind == OutputKind.STOP_AND_ALERT:
            output = system_trace.output
        elif risk.name == "STOP_AND_ALERT":
            output = RobotOutput(
                kind=OutputKind.STOP_AND_ALERT,
                chassis="STOP",
                led="RED",
                screen=f"POSSIBLE PERSON: {prediction.confidence:.2f}",
                sound="ALERT",
            )
        elif risk.name == "CONFIRM_OBJECT_FOR_TEACHER":
            output = RobotOutput(
                kind=OutputKind.OBJECT_CONFIRMED,
                chassis="STOP",
                led="BLUE",
                screen=f"OBJECT CANDIDATE: {prediction.label}",
            )
        else:
            output = RobotOutput(
                kind=OutputKind.HOLD,
                chassis="STOP",
                led="AMBER",
                screen="OBSERVING",
            )

        feedback = (
            f"output={output.kind.value}; next input must use a new sequence and frame id"
        )
        return AgentTrace(
            system=system_trace,
            prediction=prediction,
            fresh_prediction=fresh,
            prediction_streak=streak,
            risk_decision=risk,
            output=output,
            feedback=feedback,
        )
