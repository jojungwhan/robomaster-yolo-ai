import unittest
import time

import numpy as np

from lego_identification import (
    AsyncLegoIdentificationDetector,
    LegoIdentificationDetector,
    normalize_piece_label,
    parse_piece_attributes,
)


class FakeBox:
    xyxy = [np.array([10, 20, 110, 120])]
    cls = np.array([0])
    conf = np.array([0.87])


class FakeResult:
    names = {0: "Red_Rectangle_Medium"}
    boxes = [FakeBox()]


class FakeModel:
    names = FakeResult.names

    def __init__(self):
        self.calls = 0

    def predict(self, **_kwargs):
        self.calls += 1
        return [FakeResult()]


class LegoIdentificationTests(unittest.TestCase):
    def test_model_label_normalization_and_attributes(self):
        self.assertEqual(
            normalize_piece_label("Red_Rectangle_Medium"),
            "lego_piece_red_rectangle_medium",
        )
        self.assertEqual(
            parse_piece_attributes("Red_Rectangle_Medium"),
            {
                "raw_label": "Red_Rectangle_Medium",
                "color": "red",
                "shape": "rectangle",
                "size": "medium",
            },
        )

    def test_detector_caches_between_inference_intervals(self):
        model = FakeModel()
        detector = LegoIdentificationDetector(
            "unused.pt",
            interval_seconds=0.5,
            model=model,
        )
        frame = np.zeros((200, 300, 3), dtype=np.uint8)

        first = detector.detect(frame, now=1.0)
        cached = detector.detect(frame, now=1.2)
        refreshed = detector.detect(frame, now=1.6)

        self.assertEqual(model.calls, 2)
        self.assertEqual(first, cached)
        self.assertEqual(first, refreshed)
        self.assertEqual(first[0]["label"], "lego_piece_red_rectangle_medium")
        self.assertEqual(first[0]["source"], "lego_identification")

    def test_async_detector_returns_background_result(self):
        model = FakeModel()
        detector = LegoIdentificationDetector(
            "unused.pt",
            interval_seconds=0.01,
            model=model,
        )
        async_detector = AsyncLegoIdentificationDetector(detector)
        self.addCleanup(async_detector.close)
        frame = np.zeros((200, 300, 3), dtype=np.uint8)

        self.assertEqual(async_detector.detect(frame), [])
        deadline = time.monotonic() + 1
        detections = []
        while not detections and time.monotonic() < deadline:
            time.sleep(0.01)
            detections = async_detector.detect(frame)

        self.assertTrue(detections)
        self.assertEqual(
            detections[0]["label"],
            "lego_piece_red_rectangle_medium",
        )


if __name__ == "__main__":
    unittest.main()
