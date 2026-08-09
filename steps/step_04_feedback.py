"""Step 4: an output changes what the next loop should inspect."""

from fundamentals import RobotInput, RobotSystem


system = RobotSystem()
first = system.tick(RobotInput(sequence=1, distance_cm=35))
print("TICK 1 OUTPUT  ", first.output)
print("TICK 1 FEEDBACK", first.feedback)

second = system.tick(RobotInput(sequence=2, button="stop", distance_cm=90))
print("TICK 2 INPUT   ", second.input)
print("TICK 2 OUTPUT  ", second.output)
print("The loop did not assume the first scene stayed true; it read fresh input again.")
