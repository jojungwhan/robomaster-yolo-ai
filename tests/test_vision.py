from __future__ import annotations

import unittest

from fundamentals import AIPrediction, Frame, ObservationMemory, RiskPolicy
from fundamentals.vision import evaluate_prediction, measure_bright_region


class VisionAndAITest(unittest.TestCase):
    def test_computer_vision_rule_counts_pixels(self) -> None:
        frame = Frame("f1", ((0, 255), (200, 199)))
        measurement = measure_bright_region(frame, pixel_threshold=200)
        self.assertEqual(measurement.bright_pixel_count, 2)
        self.assertEqual(measurement.bright_fraction, 0.5)

    def test_duplicate_frame_does_not_increase_evidence(self) -> None:
        memory = ObservationMemory()
        prediction = AIPrediction("f1", "bottle", 0.8)
        self.assertEqual(memory.observe(prediction), (1, True))
        self.assertEqual(memory.observe(prediction), (1, False))

    def test_lower_person_threshold_only_requests_protection(self) -> None:
        decision = evaluate_prediction(AIPrediction("f1", "person", 0.36), 1)
        self.assertEqual(decision.name, "STOP_AND_ALERT")
        self.assertFalse(decision.allows_motion)

    def test_object_confirmation_requires_high_score_and_fresh_frames(self) -> None:
        prediction = AIPrediction("f3", "bottle", 0.74)
        self.assertEqual(evaluate_prediction(prediction, 2).name, "HOLD_AND_OBSERVE")
        decision = evaluate_prediction(prediction, 3)
        self.assertEqual(decision.name, "CONFIRM_OBJECT_FOR_TEACHER")
        self.assertFalse(decision.allows_motion)

    def test_threshold_values_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            RiskPolicy(person_stop_threshold=-0.1)


if __name__ == "__main__":
    unittest.main()
