from __future__ import annotations

import unittest

from fundamentals import OutputKind, RobotInput, RobotSystem


class RobotSystemTest(unittest.TestCase):
    def test_clear_input_produces_visible_status_but_no_motion(self) -> None:
        trace = RobotSystem().tick(RobotInput(1, distance_cm=100))
        self.assertEqual(trace.decision, "READY")
        self.assertEqual(trace.output.kind, OutputKind.STATUS)
        self.assertEqual(trace.output.chassis, "STOP")

    def test_near_input_stops_and_alerts(self) -> None:
        trace = RobotSystem().tick(RobotInput(1, distance_cm=20))
        self.assertEqual(trace.output.kind, OutputKind.STOP_AND_ALERT)
        self.assertEqual(trace.output.led, "RED")

    def test_stale_sequence_does_not_replace_memory(self) -> None:
        system = RobotSystem()
        first = system.tick(RobotInput(1, distance_cm=100))
        stale = system.tick(RobotInput(1, distance_cm=10))
        self.assertEqual(stale.decision, "STALE_INPUT")
        self.assertEqual(stale.memory_after, first.memory_after)

    def test_invalid_distance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RobotSystem().tick(RobotInput(1, distance_cm=-1))


if __name__ == "__main__":
    unittest.main()
