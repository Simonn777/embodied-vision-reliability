"""Fixed estimator and causal scores. This module never receives ground truth."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    area: int
    anisotropy: float


def largest_component(mask):
    """4-connected component; equal sizes choose smallest row-major pixel."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Component mask must be 2D")
    height, width = mask.shape
    visited = np.zeros_like(mask)
    best = []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        component = []
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    return np.asarray(best, dtype=np.int64).reshape(-1, 2)


def detect(image, color, color_distance=0.18):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image")
    color = np.asarray(color, dtype=np.float64)
    if color.shape != (3,) or not np.isfinite(color).all():
        raise ValueError("Color must have three finite channels")
    difference = image.astype(np.float64) / 255 - color
    mask = np.sum(difference ** 2, axis=2) <= color_distance ** 2
    component = largest_component(mask)
    if not len(component):
        return None
    mean_yx = component.mean(axis=0)
    centered = component - mean_yx
    covariance = centered.T @ centered / len(component)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.25)
    return Detection(float(mean_yx[1]), float(mean_yx[0]), len(component), float(eigenvalues[-1] / eigenvalues[0]))


def score_detection(detection, initial, history, frame):
    """Scores only observed shape and past raw predictions, never acceptance."""
    if any(timestamp >= frame for timestamp, _ in history):
        raise ValueError("Scoring may only use strictly past predictions")
    if detection is None:
        return {"static": math.inf, "temporal": math.inf, "innovation": None}
    static = abs(math.log(detection.area / initial.area))
    static += 0.25 * max(0., math.log(detection.anisotropy / initial.anisotropy))
    point = np.array([detection.x, detection.y])
    if len(history) < 2:
        innovation = 0.
    else:
        last_time, last = history[-1]
        predicted = np.asarray(last, dtype=np.float64)
        previous_time, previous = history[-2]
        if previous_time >= last_time:
            raise ValueError("Prediction history timestamps must increase")
        velocity = (predicted - previous) / (last_time - previous_time)
        predicted = predicted + (frame - last_time) * velocity
        if last_time >= frame:
            raise ValueError("Scoring may only use strictly past predictions")
        innovation = float(np.linalg.norm(point - predicted))
    temporal = static + min(innovation / math.sqrt(initial.area / math.pi), 10.)
    return {"static": static, "temporal": temporal, "innovation": innovation}


class FixedTracker:
    """Initial query is an explicit oracle input; no later ground truth input."""

    def __init__(self, first_frame, initial_query):
        frame = np.asarray(first_frame)
        query = np.asarray(initial_query, dtype=np.float64)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 initial frame")
        if query.shape != (2,) or not np.isfinite(query).all():
            raise ValueError("Initial query must contain finite x,y")
        x, y = np.rint(query).astype(int)
        if not (2 <= x < frame.shape[1] - 2 and 2 <= y < frame.shape[0] - 2):
            raise ValueError("Initial query must permit a complete 5x5 patch")
        self.color = frame[y - 2:y + 3, x - 2:x + 3].astype(np.float64).mean(axis=(0, 1)) / 255
        self.initial = detect(frame, self.color)
        if self.initial is None:
            raise ValueError("Initial frame must contain a detectable target")
        self.history = [(0, np.array([self.initial.x, self.initial.y]))]
        self.last_frame = 0

    def step(self, image, frame):
        if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer)) or frame <= self.last_frame:
            raise ValueError("Frame timestamps must increase")
        detection = detect(image, self.color)
        scores = score_detection(detection, self.initial, self.history, frame)
        if detection is not None:
            self.history.append((frame, np.array([detection.x, detection.y])))
            self.history = self.history[-2:]
        self.last_frame = frame
        return detection, scores
