"""Step 1: a sensor value enters; visible robot state comes out."""

from fundamentals import RobotInput, RobotSystem


system = RobotSystem()
trace = system.tick(RobotInput(sequence=1, button="ready", distance_cm=120))

print("INPUT   ", trace.input)
print("OUTPUT  ", trace.output)
print("RULE    ", trace.decision)
print("QUESTION: Which part came from the world, and which part changed the world?")
