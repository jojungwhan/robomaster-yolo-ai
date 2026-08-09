"""Step 2: Python checks and names raw input values."""

from fundamentals.system import RobotInput, process_input


for sequence, distance in enumerate((None, 120, 35), start=1):
    raw = RobotInput(sequence=sequence, distance_cm=distance)
    processed = process_input(raw)
    print(f"raw={distance!r:>4} cm -> processed={processed.distance_band}")

print("PROCESS is a written rule here, so it is automation rather than learned AI.")
