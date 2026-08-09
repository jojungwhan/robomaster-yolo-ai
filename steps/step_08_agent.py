"""Step 8: observe, remember, decide, output, and observe again."""

from fundamentals import AIPrediction, RobotAgent, RobotInput


agent = RobotAgent()
predictions = (
    AIPrediction("frame-1", "bottle", 0.74),
    AIPrediction("frame-2", "bottle", 0.73),
    AIPrediction("frame-3", "bottle", 0.72),
    AIPrediction("frame-4", "person", 0.36),
)

for sequence, prediction in enumerate(predictions, start=1):
    trace = agent.tick(
        RobotInput(sequence=sequence, distance_cm=100, frame_id=prediction.frame_id),
        prediction,
    )
    print(
        f"tick={sequence} input={prediction.label}:{prediction.confidence:.2f} "
        f"memory_streak={trace.prediction_streak} decision={trace.risk_decision.name} "
        f"output={trace.output.kind.value}/{trace.output.chassis}"
    )

print("This is a hybrid agent: an AI observation plus Python memory, rules, and safety.")
