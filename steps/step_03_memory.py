"""Step 3: working memory lets this tick remember the previous tick."""

from fundamentals import RobotInput, RobotSystem


system = RobotSystem()
for sequence, distance in ((1, 100), (2, 80), (2, 20), (3, 45)):
    trace = system.tick(RobotInput(sequence=sequence, distance_cm=distance))
    print(
        f"seq={sequence} before={trace.memory_before} "
        f"decision={trace.decision} after={trace.memory_after}"
    )

print("The duplicate sequence was stale, so it did not overwrite working memory.")
