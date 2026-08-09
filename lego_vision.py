"""OpenCV LEGO candidate, semantic-pattern, and ArUco-marker detection.

Color/contour detections are intended for controlled demos. For autonomous approach,
an ArUco marker attached to the LEGO build is the preferred target.
"""

import cv2
import numpy as np


HSV_RANGES = {
    "red": [((0, 110, 70), (10, 255, 255)), ((170, 110, 70), (179, 255, 255))],
    "orange": [((10, 120, 90), (24, 255, 255))],
    "yellow": [((24, 100, 100), (36, 255, 255))],
    "green": [((38, 70, 60), (85, 255, 255))],
    "blue": [((90, 80, 60), (130, 255, 255))],
    "purple": [((132, 60, 55), (165, 255, 255))],
}

HELP_SIGNAL_LABEL = "lego_signal_help_needed"


class ConsecutiveDetectionGate:
    """Confirm a visual signal across frames and release it after it disappears."""

    def __init__(self, required_frames=3, release_frames=8):
        if required_frames < 1 or release_frames < 1:
            raise ValueError("Frame counts must be positive.")
        self.required_frames = required_frames
        self.release_frames = release_frames
        self.reset()

    def reset(self):
        self._hits = 0
        self._misses = 0
        self.confirmed = False

    def update(self, detected):
        newly_confirmed = False
        if detected:
            self._hits += 1
            self._misses = 0
            if not self.confirmed and self._hits >= self.required_frames:
                self.confirmed = True
                newly_confirmed = True
        else:
            self._hits = 0
            self._misses += 1
            if self._misses >= self.release_frames:
                self.confirmed = False

        return self.confirmed, newly_confirmed


def _scaled_box(box, inverse_scale):
    x, y, width, height = box
    return tuple(
        round(value * inverse_scale)
        for value in (x, y, x + width, y + height)
    )


def _resize_for_detection(frame, working_width):
    frame_height, frame_width = frame.shape[:2]
    scale = min(1.0, working_width / frame_width)
    if scale < 1.0:
        working = cv2.resize(
            frame,
            (round(frame_width * scale), round(frame_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = frame
    return working, scale


def _hsv_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(
            hsv,
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8),
        )
    return mask


def _cross_template(size=96, stroke_fraction=0.30):
    template = np.zeros((size, size), dtype=np.uint8)
    half_stroke = max(2, round(size * stroke_fraction / 2))
    center = size // 2
    template[:, center - half_stroke : center + half_stroke] = 1
    template[center - half_stroke : center + half_stroke, :] = 1
    return template


def _red_cross_similarity(candidate):
    """Return a template score for a roughly upright plus-shaped red region."""
    normalized_size = 96
    normalized = cv2.resize(
        candidate,
        (normalized_size, normalized_size),
        interpolation=cv2.INTER_NEAREST,
    )
    normalized = (normalized > 0).astype(np.uint8)
    best_score = 0.0
    best_arm_coverage = 0.0
    best_corner_occupancy = 1.0

    center = (normalized_size / 2, normalized_size / 2)
    for stroke_fraction in (
        0.12,
        0.15,
        0.18,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ):
        base_template = _cross_template(normalized_size, stroke_fraction)
        for angle in (-20, -15, -10, -5, 0, 5, 10, 15, 20):
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            template = cv2.warpAffine(
                base_template,
                matrix,
                (normalized_size, normalized_size),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
            template = (template > 0).astype(np.uint8)
            arm_pixels = template == 1
            corner_pixels = template == 0
            arm_coverage = float(normalized[arm_pixels].mean())
            corner_occupancy = float(normalized[corner_pixels].mean())
            intersection = np.logical_and(normalized, template).sum()
            union = np.logical_or(normalized, template).sum()
            iou = float(intersection / max(union, 1))
            score = (
                0.55 * iou
                + 0.30 * arm_coverage
                + 0.15 * (1 - corner_occupancy)
            )
            if score > best_score:
                best_score = score
                best_arm_coverage = arm_coverage
                best_corner_occupancy = corner_occupancy

    return best_score, best_arm_coverage, best_corner_occupancy


def detect_red_cross_signal(frame, working_width=960):
    """Detect a red LEGO plus/cross used as a semantic help-needed signal.

    The detector deliberately requires a near-square, concave plus shape. A plain red
    square or a single red rectangular brick is rejected. Confirm detections over
    multiple frames with :class:`ConsecutiveDetectionGate` before taking action.
    """
    working, scale = _resize_for_detection(frame, working_width)
    hsv = cv2.cvtColor(cv2.GaussianBlur(working, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = _hsv_mask(hsv, HSV_RANGES["red"])

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    working_area = working.shape[0] * working.shape[1]
    min_area = max(220, working_area * 0.00035)
    max_area = working_area * 0.30
    inverse_scale = 1.0 / scale
    detections = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if min(width, height) < 28:
            continue
        aspect_ratio = width / max(height, 1)
        if not 0.68 <= aspect_ratio <= 1.47:
            continue

        fill_ratio = area / max(width * height, 1)
        if not 0.15 <= fill_ratio <= 0.82:
            continue

        candidate = mask[y : y + height, x : x + width]
        score, arm_coverage, corner_occupancy = _red_cross_similarity(candidate)
        if score < 0.70 or arm_coverage < 0.65 or corner_occupancy > 0.28:
            continue

        confidence = min(0.98, max(0.70, score))
        detections.append(
            {
                "box": _scaled_box((x, y, width, height), inverse_scale),
                "label": HELP_SIGNAL_LABEL,
                "confidence": confidence,
                "source": "opencv_pattern",
                "pattern": "red_cross",
                "signal": "help_needed",
            }
        )

    return detections


def detect_colored_lego_candidates(frame, working_width=960):
    """Return rectangular, saturated-color regions that may be LEGO bricks."""
    working, scale = _resize_for_detection(frame, working_width)

    blurred = cv2.GaussianBlur(working, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    working_area = working.shape[0] * working.shape[1]
    min_area = max(90, working_area * 0.00012)
    max_area = working_area * 0.12
    kernel = np.ones((3, 3), np.uint8)
    inverse_scale = 1.0 / scale
    detections = []

    for color, ranges in HSV_RANGES.items():
        mask = _hsv_mask(hsv, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if not min_area <= area <= max_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width < 8 or height < 8:
                continue
            aspect_ratio = width / height
            rectangularity = area / max(width * height, 1)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / max(hull_area, 1)
            if not 0.3 <= aspect_ratio <= 4.5:
                continue
            if rectangularity < 0.55 or solidity < 0.75:
                continue

            confidence = min(0.92, 0.35 + 0.35 * rectangularity + 0.25 * solidity)
            detections.append(
                {
                    "box": _scaled_box((x, y, width, height), inverse_scale),
                    "label": f"lego_{color}",
                    "confidence": confidence,
                    "source": "opencv_color",
                }
            )

    return detections


def detect_aruco_lego_markers(frame):
    """Detect 4x4 ArUco IDs attached to LEGO targets."""
    if not hasattr(cv2, "aruco"):
        return []

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = None
    if hasattr(aruco, "DetectorParameters"):
        try:
            parameters = aruco.DetectorParameters()
        except Exception:
            parameters = None
    if parameters is None and hasattr(aruco, "DetectorParameters_create"):
        parameters = aruco.DetectorParameters_create()
    if parameters is None:
        return []

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        corners, ids, _ = aruco.detectMarkers(
            frame,
            dictionary,
            parameters=parameters,
        )

    if ids is None:
        return []

    detections = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        points = marker_corners.reshape(-1, 2)
        x1, y1 = np.floor(points.min(axis=0)).astype(int)
        x2, y2 = np.ceil(points.max(axis=0)).astype(int)
        detections.append(
            {
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "label": f"lego_marker_{int(marker_id)}",
                "confidence": 0.99,
                "source": "aruco",
            }
        )
    return detections


def detect_lego(frame):
    return (
        detect_aruco_lego_markers(frame)
        + detect_colored_lego_candidates(frame)
        + detect_red_cross_signal(frame)
    )


def generate_marker(marker_id=0, size_pixels=600):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV installation does not include cv2.aruco.")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size_pixels)
    marker = np.zeros((size_pixels, size_pixels), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, size_pixels, marker, 1)
    return marker
