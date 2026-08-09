"""Step 7: compare written pixel rules with a learned model prediction."""

from fundamentals import AIPrediction, Frame, ObservationMemory, RiskPolicy, measure_bright_region
from fundamentals.vision import evaluate_prediction


frame = Frame("frame-1", ((10, 10, 240), (10, 230, 250), (5, 20, 30)))
measurement = measure_bright_region(frame, pixel_threshold=200)
print("COMPUTER VISION RULE", measurement)

policy = RiskPolicy(person_stop_threshold=0.30, object_confirm_threshold=0.70)
memory = ObservationMemory()
for prediction in (
    AIPrediction("frame-1", "person", 0.36),
    AIPrediction("frame-2", "bottle", 0.74),
    AIPrediction("frame-3", "bottle", 0.76),
    AIPrediction("frame-4", "bottle", 0.72),
):
    streak, fresh = memory.observe(prediction)
    decision = evaluate_prediction(prediction, streak, policy)
    print("AI OBSERVATION", prediction, "fresh=", fresh, "streak=", streak, "->", decision)

print("A lower person threshold produced protection, never permission to approach.")
