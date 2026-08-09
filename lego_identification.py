"""Adapter for the vsmidhun21/Lego-Identification Ultralytics model."""

import hashlib
import queue
import re
import threading
import time
from pathlib import Path

from ultralytics import YOLO


PINNED_MODEL_NAME = "FinalCoShSi.pt"
PINNED_MODEL_SHA256 = (
    "87591257D011CC7409CFF14BABF28A1D15402AB521E75F3D10BF5F7A1E013CF6"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalize_piece_label(raw_label):
    normalized = re.sub(r"[^a-z0-9]+", "_", str(raw_label).lower()).strip("_")
    return f"lego_piece_{normalized}"


def parse_piece_attributes(raw_label):
    parts = [part for part in str(raw_label).split("_") if part]
    attributes = {"raw_label": str(raw_label)}
    if parts:
        attributes["color"] = parts[0].lower()
    if len(parts) >= 3:
        attributes["shape"] = "_".join(parts[1:-1]).lower()
    if len(parts) >= 2:
        attributes["size"] = parts[-1].lower()
    return attributes


class LegoIdentificationDetector:
    """Run the third-party LEGO detector at a limited cadence and cache its boxes."""

    def __init__(
        self,
        model_path,
        confidence=0.30,
        inference_size=640,
        interval_seconds=0.45,
        model=None,
    ):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.inference_size = inference_size
        self.interval_seconds = interval_seconds
        self._last_inference = float("-inf")
        self._cached = []

        if not 0 <= confidence <= 1:
            raise ValueError("LEGO model confidence must be between 0 and 1.")
        if inference_size < 160:
            raise ValueError("LEGO inference size must be at least 160.")
        if interval_seconds < 0:
            raise ValueError("LEGO model interval cannot be negative.")

        if model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(self.model_path)
            if self.model_path.name == PINNED_MODEL_NAME:
                actual_hash = sha256_file(self.model_path)
                if actual_hash != PINNED_MODEL_SHA256:
                    raise RuntimeError(
                        f"Unexpected SHA256 for {self.model_path}: {actual_hash}"
                    )
            self.model = YOLO(str(self.model_path))
        else:
            self.model = model

        names = getattr(self.model, "names", {})
        self.class_count = len(names)

    @property
    def summary(self):
        return f"ON ({self.class_count} classes, {self.interval_seconds:.2f}s cadence)"

    def detect(self, frame, now=None, force=False):
        now = time.monotonic() if now is None else now
        if not force and now - self._last_inference < self.interval_seconds:
            return [dict(item) for item in self._cached]

        results = self.model.predict(
            source=frame,
            verbose=False,
            conf=self.confidence,
            imgsz=self.inference_size,
            max_det=100,
        )
        detections = []
        for result in results:
            names = getattr(result, "names", self.model.names)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                raw_label = names[class_id]
                detection = {
                    "box": (x1, y1, x2, y2),
                    "label": normalize_piece_label(raw_label),
                    "confidence": float(box.conf[0]),
                    "source": "lego_identification",
                }
                detection.update(parse_piece_attributes(raw_label))
                detections.append(detection)

        self._last_inference = now
        self._cached = detections
        return [dict(item) for item in detections]


class AsyncLegoIdentificationDetector:
    """Keep the preview responsive while the slower LEGO model runs off-thread."""

    def __init__(self, detector, max_cache_age_seconds=None):
        self.detector = detector
        self.max_cache_age_seconds = max_cache_age_seconds or max(
            1.5,
            detector.interval_seconds * 2,
        )
        self._requests = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._latest = []
        self._latest_at = float("-inf")
        self._last_submit = float("-inf")
        self._generation = 0
        self._error = None
        self._worker = threading.Thread(
            target=self._run,
            name="lego-identification",
            daemon=True,
        )
        self._worker.start()

    @property
    def summary(self):
        return f"{self.detector.summary}, async"

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self._requests.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                return

            self._busy.set()
            try:
                detections = self.detector.detect(frame, force=True)
                completed_at = time.monotonic()
                with self._lock:
                    self._generation += 1
                    observation_id = f"lego_identification:{self._generation}"
                    self._latest = [
                        {**item, "observation_id": observation_id}
                        for item in detections
                    ]
                    self._latest_at = completed_at
            except Exception as error:
                with self._lock:
                    self._error = error
                return
            finally:
                self._busy.clear()

    def detect(self, frame, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            error = self._error
            latest = [dict(item) for item in self._latest]
            latest_at = self._latest_at
        if error is not None:
            raise RuntimeError("Background LEGO inference failed") from error

        can_submit = (
            now - self._last_submit >= self.detector.interval_seconds
            and not self._busy.is_set()
            and self._requests.empty()
        )
        if can_submit:
            try:
                self._requests.put_nowait(frame.copy())
                self._last_submit = now
            except queue.Full:
                pass

        if now - latest_at > self.max_cache_age_seconds:
            return []
        return latest

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=2)
