"""Seeded toy sequences. Ground truth stays on the rendering/evaluation side."""

import hashlib
from dataclasses import dataclass

import numpy as np

CONDITIONS = ("clean", "occlusion", "appearance", "abrupt_motion")
MASTER_SEED = 20261016
SPLIT_SIZES = {"fit": 80, "dev": 20, "cal": 20, "test": 40}


def derived_seed(master, sequence_id, stream, frame=-1):
    key = f"{master}|{sequence_id:03d}|{stream}|{frame}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def tie_key(sequence_id, frame):
    return hashlib.sha256(f"{sequence_id:03d}|{frame}".encode("utf-8")).hexdigest()


def split_manifest(smoke=False):
    sizes = {"fit": 2, "dev": 2, "cal": 2, "test": 4} if smoke else SPLIT_SIZES
    first = 10000 if smoke else 0
    records = []
    for partition, count in sizes.items():
        for sequence_id in range(first, first + count):
            records.append({"sequence_id": sequence_id, "partition": partition})
        first += count
    return records


@dataclass(frozen=True)
class Scene:
    sequence_id: int
    radius: int
    initial: np.ndarray
    velocity: np.ndarray
    color: np.ndarray
    distractor: np.ndarray


def make_scene(sequence_id, master=MASTER_SEED):
    rng = np.random.default_rng(derived_seed(master, sequence_id, "scene"))
    radius = int(rng.integers(5, 9))
    initial = rng.uniform(28, 68, size=2)
    angle = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(0.6, 1.2)
    velocity = speed * np.array([np.cos(angle), np.sin(angle)])
    color = np.array([0.2, 0.75, 0.25]) + rng.uniform(-0.03, 0.03, size=3)
    corners = np.array([[18., 18.], [78., 78.], [18., 78.], [78., 18.]])
    distractor = corners[np.argmax(np.sum((corners - initial) ** 2, axis=1))]
    return Scene(sequence_id, radius, initial, velocity, color, distractor)


def trajectory(scene, condition, frames=32):
    if condition not in CONDITIONS:
        raise ValueError("Unknown rendering condition")
    position = scene.initial.copy()
    velocity = scene.velocity.copy()
    points = []
    for frame in range(frames):
        if frame:
            if condition == "abrupt_motion" and frame == 16:
                velocity = np.array([-velocity[1], velocity[0]]) * 2
            position = position + velocity
            for axis in (0, 1):
                if position[axis] < 16:
                    position[axis] = 32 - position[axis]
                    velocity[axis] *= -1
                elif position[axis] > 80:
                    position[axis] = 160 - position[axis]
                    velocity[axis] *= -1
        points.append(position.copy())
    return np.asarray(points)


def render_frame(scene, point, frame, condition, size=96, master=MASTER_SEED):
    """Return RGB uint8. Same scene/frame noise is paired across conditions."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown rendering condition")
    if size != 96:
        raise ValueError("Protocol 0001 fixes the image size at 96")
    rng = np.random.default_rng(derived_seed(master, scene.sequence_id, "background", frame))
    image = np.clip(0.12 + rng.normal(0, 0.015, (size, size, 3)), 0, 1)
    yy, xx = np.ogrid[:size, :size]
    target = (xx - point[0]) ** 2 + (yy - point[1]) ** 2 <= scene.radius ** 2
    color = scene.color
    if condition == "appearance" and frame >= 8:
        color = 0.45 * scene.color + 0.55 * 0.12
    image[target] = color
    if condition == "appearance" and frame >= 8:
        distractor = ((xx - scene.distractor[0]) ** 2 + (yy - scene.distractor[1]) ** 2 <= scene.radius ** 2)
        image[distractor] = scene.color
    if condition == "occlusion" and 8 <= frame <= 23:
        right = point[0] + (scene.radius + 2 if 14 <= frame <= 17 else 0.35 * scene.radius)
        occluder = ((xx >= point[0] - scene.radius - 2) & (xx <= right) &
                    (yy >= point[1] - scene.radius - 2) & (yy <= point[1] + scene.radius + 2))
        image[occluder] = [0.2, 0.22, 0.25]
    return np.rint(image * 255).astype(np.uint8)
