"""
Shared helpers for running MediaPipe Pose and building gesture/structure primitives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


LMS = mp.solutions.pose.PoseLandmark


# -----------------------------------------------------------------------------
# Data containers


@dataclass
class GestureCurve:
    name: str
    points: np.ndarray  # (N, 2)


@dataclass
class OrientedBox:
    name: str
    center: np.ndarray  # (3,)
    axes: np.ndarray  # (3, 3)
    size: Tuple[float, float, float]

    def corners(self) -> np.ndarray:
        half_size = np.array(self.size) / 2.0
        corners = []
        for dx in (-1.0, 1.0):
            for dy in (-1.0, 1.0):
                for dz in (-1.0, 1.0):
                    offset = half_size * np.array([dx, dy, dz])
                    corners.append(self.center + self.axes @ offset)
        return np.array(corners)


@dataclass
class LimbSegment:
    name: str
    start: np.ndarray  # (3,)
    end: np.ndarray  # (3,)
    radius: float


@dataclass
class OrientedBox2D:
    name: str
    center: np.ndarray  # (2,)
    axes: np.ndarray  # (2, 2)
    size: Tuple[float, float]  # (width, height) along x/y axes

    def corners(self) -> np.ndarray:
        half_size = np.array(self.size) / 2.0
        corners = []
        for dx in (-1.0, 1.0):
            for dy in (-1.0, 1.0):
                offset = half_size * np.array([dx, dy])
                corners.append(self.center + self.axes @ offset)
        return np.array(corners)


@dataclass
class LimbSegment2D:
    name: str
    start: np.ndarray  # (2,)
    end: np.ndarray  # (2,)
    thickness: float


# -----------------------------------------------------------------------------
# Pose helpers


def run_pose_estimation(image_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read image and return image, 2D landmarks, and MediaPipe world landmarks."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks or not results.pose_world_landmarks:
        raise RuntimeError("MediaPipe Pose did not detect a person in the image.")

    h, w = image.shape[:2]
    k2d = np.array(
        [
            [lm.x * w, lm.y * h, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ],
        dtype=np.float32,
    )
    k3d = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark],
        dtype=np.float32,
    )
    return image, k2d, k3d


def draw_landmark_labels(
    image: np.ndarray,
    landmarks_2d: np.ndarray,
    landmark_names: Sequence[Tuple[LMS, str]],
    radius: int = 6,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw selected landmarks with labels on an image copy."""
    annotated = image.copy()
    for enum_member, label in landmark_names:
        idx = enum_member.value
        x, y, visibility = landmarks_2d[idx]
        if visibility < 0.2:
            continue
        center = (int(round(x)), int(round(y)))
        cv2.circle(annotated, center, radius, (0, 80, 255), -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            label,
            (center[0] + radius + 2, center[1] - radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return annotated


# -----------------------------------------------------------------------------
# Gesture helpers


def smooth_curve(points: np.ndarray, iterations: int = 3) -> np.ndarray:
    if len(points) < 3:
        return points.copy()

    curve = np.asarray(points, dtype=np.float32)
    for _ in range(iterations):
        q = curve[:-1] * 0.75 + curve[1:] * 0.25
        r = curve[:-1] * 0.25 + curve[1:] * 0.75
        curve = np.vstack([np.column_stack([q, r]).reshape(-1, 2), curve[-1]])
    return curve


def build_gesture_curves(landmarks_2d: np.ndarray, iterations: int = 3) -> List[GestureCurve]:
    chains = {
        "line_of_action": [
            LMS.LEFT_ANKLE,
            LMS.LEFT_KNEE,
            LMS.LEFT_HIP,
            LMS.RIGHT_HIP,
            LMS.LEFT_SHOULDER,
            LMS.NOSE,
        ],
        "right_arm": [
            LMS.RIGHT_SHOULDER,
            LMS.RIGHT_ELBOW,
            LMS.RIGHT_WRIST,
            LMS.RIGHT_INDEX,
        ],
        "left_arm": [
            LMS.LEFT_SHOULDER,
            LMS.LEFT_ELBOW,
            LMS.LEFT_WRIST,
            LMS.LEFT_INDEX,
        ],
        "support_leg": [
            LMS.LEFT_HIP,
            LMS.LEFT_KNEE,
            LMS.LEFT_ANKLE,
            LMS.LEFT_FOOT_INDEX,
        ],
        "gesture_leg": [
            LMS.RIGHT_HIP,
            LMS.RIGHT_KNEE,
            LMS.RIGHT_ANKLE,
            LMS.RIGHT_FOOT_INDEX,
        ],
    }

    curves: List[GestureCurve] = []
    for name, ids in chains.items():
        pts = np.array([landmarks_2d[idx.value, :2] for idx in ids], dtype=np.float32)
        curves.append(GestureCurve(name=name, points=smooth_curve(pts, iterations)))
    return curves


def draw_gesture_overlay(
    image: np.ndarray, curves: Sequence[GestureCurve], thickness: int = 3
) -> np.ndarray:
    overlay = image.copy()
    colors = {
        "line_of_action": (0, 200, 50),
        "right_arm": (200, 80, 0),
        "left_arm": (0, 120, 240),
        "support_leg": (140, 0, 200),
        "gesture_leg": (200, 0, 120),
    }

    for curve in curves:
        color = colors.get(curve.name, (0, 180, 0))
        pts = curve.points.astype(int)
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(overlay, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
    return overlay


# -----------------------------------------------------------------------------
# Structure helpers


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return vec
    return vec / norm


def _frame_from_direction(x_dir: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    x_axis = _normalize(x_dir)
    up_hint = _normalize(up_hint)
    z_axis = _normalize(np.cross(x_axis, up_hint))
    if np.linalg.norm(z_axis) < 1e-6:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def compute_oriented_boxes(landmarks_3d: np.ndarray) -> List[OrientedBox]:
    lhip = landmarks_3d[LMS.LEFT_HIP.value]
    rhip = landmarks_3d[LMS.RIGHT_HIP.value]
    lshoulder = landmarks_3d[LMS.LEFT_SHOULDER.value]
    rshoulder = landmarks_3d[LMS.RIGHT_SHOULDER.value]
    nose = landmarks_3d[LMS.NOSE.value]
    left_ear = landmarks_3d[LMS.LEFT_EAR.value]
    right_ear = landmarks_3d[LMS.RIGHT_EAR.value]

    hip_center = _midpoint(lhip, rhip)
    shoulder_center = _midpoint(lshoulder, rshoulder)
    spine_dir = shoulder_center - hip_center
    hip_width = np.linalg.norm(rhip - lhip)
    shoulder_width = np.linalg.norm(rshoulder - lshoulder)
    torso_height = np.linalg.norm(spine_dir)

    pelvis_axes = _frame_from_direction(rhip - lhip, shoulder_center - hip_center)
    pelvis_size = (
        max(hip_width * 1.1, 1e-3),
        max(torso_height * 0.45, 1e-3),
        max(hip_width * 0.7, 1e-3),
    )
    pelvis = OrientedBox("pelvis", hip_center, pelvis_axes, pelvis_size)

    ribcage_center = hip_center + spine_dir * 0.6
    ribcage_axes = _frame_from_direction(rshoulder - lshoulder, nose - ribcage_center)
    ribcage_size = (
        max(shoulder_width * 1.15, 1e-3),
        max(torso_height * 0.65, 1e-3),
        max(shoulder_width * 0.75, 1e-3),
    )
    ribcage = OrientedBox("ribcage", ribcage_center, ribcage_axes, ribcage_size)

    head_center = nose + (nose - ribcage_center) * 0.2
    ear_width = np.linalg.norm(right_ear - left_ear) if np.any(right_ear) else shoulder_width * 0.4
    head_height = torso_height * 0.35 if torso_height > 1e-3 else hip_width * 0.8
    head_axes = _frame_from_direction(right_ear - left_ear, nose - ribcage_center)
    head_size = (
        max(ear_width * 1.1, 1e-3),
        max(head_height, 1e-3),
        max(ear_width * 0.9, 1e-3),
    )
    head = OrientedBox("head", head_center, head_axes, head_size)

    return [pelvis, ribcage, head]


def compute_limb_segments(landmarks_3d: np.ndarray) -> List[LimbSegment]:
    limb_pairs = {
        "left_upper_arm": (LMS.LEFT_SHOULDER, LMS.LEFT_ELBOW),
        "left_forearm": (LMS.LEFT_ELBOW, LMS.LEFT_WRIST),
        "right_upper_arm": (LMS.RIGHT_SHOULDER, LMS.RIGHT_ELBOW),
        "right_forearm": (LMS.RIGHT_ELBOW, LMS.RIGHT_WRIST),
        "left_thigh": (LMS.LEFT_HIP, LMS.LEFT_KNEE),
        "left_calf": (LMS.LEFT_KNEE, LMS.LEFT_ANKLE),
        "right_thigh": (LMS.RIGHT_HIP, LMS.RIGHT_KNEE),
        "right_calf": (LMS.RIGHT_KNEE, LMS.RIGHT_ANKLE),
    }

    segments: List[LimbSegment] = []
    for name, (a_idx, b_idx) in limb_pairs.items():
        start = landmarks_3d[a_idx.value]
        end = landmarks_3d[b_idx.value]
        length = np.linalg.norm(end - start)
        radius = max(length * 0.1, 1e-3)
        segments.append(LimbSegment(name, start, end, radius))
    return segments


def export_structure_json(
    boxes: Sequence[OrientedBox], limbs: Sequence[LimbSegment], output_path: Path
) -> None:
    payload = {
        "boxes": [
            {
                "name": box.name,
                "center": box.center.tolist(),
                "axes": box.axes.tolist(),
                "size": list(box.size),
                "corners": box.corners().tolist(),
            }
            for box in boxes
        ],
        "limbs": [
            {
                "name": limb.name,
                "start": limb.start.tolist(),
                "end": limb.end.tolist(),
                "radius": limb.radius,
            }
            for limb in limbs
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2))


def export_structure_obj(
    boxes: Sequence[OrientedBox], limbs: Sequence[LimbSegment], output_path: Path
) -> None:
    vertices: List[np.ndarray] = []
    faces: List[Tuple[int, int, int]] = []
    lines: List[Tuple[int, int]] = []

    def add_vertex(pt: np.ndarray) -> int:
        vertices.append(pt)
        return len(vertices)

    for box in boxes:
        corner_indices = [add_vertex(corner) for corner in box.corners()]
        corners_by_sign = {}
        i = 0
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    corners_by_sign[(dx, dy, dz)] = corner_indices[i]
                    i += 1
        faces.extend(
            [
                (corners_by_sign[(1, -1, -1)], corners_by_sign[(1, -1, 1)], corners_by_sign[(1, 1, 1)]),
                (corners_by_sign[(1, -1, -1)], corners_by_sign[(1, 1, 1)], corners_by_sign[(1, 1, -1)]),
                (corners_by_sign[(-1, -1, -1)], corners_by_sign[(-1, 1, 1)], corners_by_sign[(-1, -1, 1)]),
                (corners_by_sign[(-1, -1, -1)], corners_by_sign[(-1, 1, -1)], corners_by_sign[(-1, 1, 1)]),
                (corners_by_sign[(-1, 1, -1)], corners_by_sign[(1, 1, 1)], corners_by_sign[(1, 1, -1)]),
                (corners_by_sign[(-1, 1, -1)], corners_by_sign[(-1, 1, 1)], corners_by_sign[(1, 1, 1)]),
                (corners_by_sign[(-1, -1, -1)], corners_by_sign[(1, -1, -1)], corners_by_sign[(1, -1, 1)]),
                (corners_by_sign[(-1, -1, -1)], corners_by_sign[(1, -1, 1)], corners_by_sign[(-1, -1, 1)]),
                (corners_by_sign[(-1, -1, 1)], corners_by_sign[(1, -1, 1)], corners_by_sign[(1, 1, 1)]),
                (corners_by_sign[(-1, -1, 1)], corners_by_sign[(1, 1, 1)], corners_by_sign[(-1, 1, 1)]),
                (corners_by_sign[(-1, -1, -1)], corners_by_sign[(1, 1, -1)], corners_by_sign[(1, -1, -1)]),
                (corners_by_sign[(-1, -1, -1)], corners_by_sign[(-1, 1, -1)], corners_by_sign[(1, 1, -1)]),
            ]
        )

    for limb in limbs:
        start_idx = add_vertex(limb.start)
        end_idx = add_vertex(limb.end)
        lines.append((start_idx, end_idx))

    with output_path.open("w", encoding="ascii") as fh:
        fh.write("# Gesture structure export\n")
        for v in vertices:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for a, b, c in faces:
            fh.write(f"f {a} {b} {c}\n")
        for a, b in lines:
            fh.write(f"l {a} {b}\n")


# -----------------------------------------------------------------------------
# Structure helpers (2D overlays)


def compute_oriented_boxes_2d(landmarks_2d: np.ndarray) -> List[OrientedBox2D]:
    lhip = landmarks_2d[LMS.LEFT_HIP.value, :2]
    rhip = landmarks_2d[LMS.RIGHT_HIP.value, :2]
    lshoulder = landmarks_2d[LMS.LEFT_SHOULDER.value, :2]
    rshoulder = landmarks_2d[LMS.RIGHT_SHOULDER.value, :2]
    nose = landmarks_2d[LMS.NOSE.value, :2]
    left_ear = landmarks_2d[LMS.LEFT_EAR.value, :2]
    right_ear = landmarks_2d[LMS.RIGHT_EAR.value, :2]

    hip_center = _midpoint(lhip, rhip)
    shoulder_center = _midpoint(lshoulder, rshoulder)
    spine_dir = shoulder_center - hip_center
    hip_vec = rhip - lhip
    hip_width = np.linalg.norm(hip_vec)
    shoulder_width = np.linalg.norm(rshoulder - lshoulder)
    torso_height = np.linalg.norm(spine_dir)

    def _frame2d(x_dir: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
        x_axis = _normalize(x_dir)
        y_axis = _normalize(y_hint)
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
        # Ensure orthogonality
        y_axis = _normalize(y_axis - np.dot(y_axis, x_axis) * x_axis)
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
        return np.stack([x_axis, y_axis], axis=1)

    pelvis_axes = _frame2d(hip_vec, spine_dir)
    pelvis_size = (
        max(hip_width * 1.1, 1e-3),
        max(torso_height * 0.45, 1e-3),
    )
    pelvis = OrientedBox2D("pelvis", hip_center, pelvis_axes, pelvis_size)

    ribcage_center = hip_center + spine_dir * 0.6
    ribcage_axes = _frame2d(rshoulder - lshoulder, nose - ribcage_center)
    ribcage_size = (
        max(shoulder_width * 1.15, 1e-3),
        max(torso_height * 0.65, 1e-3),
    )
    ribcage = OrientedBox2D("ribcage", ribcage_center, ribcage_axes, ribcage_size)

    head_center = nose + (nose - ribcage_center) * 0.2
    ear_width = np.linalg.norm(right_ear - left_ear)
    if not np.isfinite(ear_width) or ear_width < 1e-3:
        ear_width = shoulder_width * 0.4
    head_height = torso_height * 0.35 if torso_height > 1e-3 else hip_width * 0.8
    head_axes = _frame2d(right_ear - left_ear, nose - ribcage_center)
    head_size = (
        max(ear_width * 1.1, 1e-3),
        max(head_height, 1e-3),
    )
    head = OrientedBox2D("head", head_center, head_axes, head_size)

    return [pelvis, ribcage, head]


def compute_limb_segments_2d(landmarks_2d: np.ndarray) -> List[LimbSegment2D]:
    limb_pairs = {
        "left_upper_arm": (LMS.LEFT_SHOULDER, LMS.LEFT_ELBOW),
        "left_forearm": (LMS.LEFT_ELBOW, LMS.LEFT_WRIST),
        "right_upper_arm": (LMS.RIGHT_SHOULDER, LMS.RIGHT_ELBOW),
        "right_forearm": (LMS.RIGHT_ELBOW, LMS.RIGHT_WRIST),
        "left_thigh": (LMS.LEFT_HIP, LMS.LEFT_KNEE),
        "left_calf": (LMS.LEFT_KNEE, LMS.LEFT_ANKLE),
        "right_thigh": (LMS.RIGHT_HIP, LMS.RIGHT_KNEE),
        "right_calf": (LMS.RIGHT_KNEE, LMS.RIGHT_ANKLE),
    }
    segments: List[LimbSegment2D] = []
    for name, (a_idx, b_idx) in limb_pairs.items():
        start = landmarks_2d[a_idx.value, :2]
        end = landmarks_2d[b_idx.value, :2]
        length = np.linalg.norm(end - start)
        thickness = max(length * 0.12, 3.0)
        segments.append(LimbSegment2D(name, start, end, thickness))
    return segments


def draw_structure_overlay(
    image: np.ndarray,
    boxes_2d: Sequence[OrientedBox2D],
    limbs_2d: Sequence[LimbSegment2D],
    landmarks_2d: np.ndarray,
    label_landmarks: bool = False,
    landmark_names: Sequence[Tuple[LMS, str]] | None = None,
) -> np.ndarray:
    overlay = image.copy()

    if label_landmarks and landmark_names:
        overlay = draw_landmark_labels(overlay, landmarks_2d, landmark_names)

    box_colors = {
        "pelvis": (255, 120, 0),
        "ribcage": (0, 180, 255),
        "head": (180, 80, 255),
    }
    for box in boxes_2d:
        color = box_colors.get(box.name, (0, 200, 200))
        corners = box.corners().astype(int)
        cv2.polylines(overlay, [corners], True, color, 2, cv2.LINE_AA)
        # draw diagonals for clarity
        cv2.line(overlay, tuple(corners[0]), tuple(corners[2]), color, 1, cv2.LINE_AA)
        cv2.line(overlay, tuple(corners[1]), tuple(corners[3]), color, 1, cv2.LINE_AA)

    limb_color = (60, 255, 120)
    for limb in limbs_2d:
        start = tuple(np.round(limb.start).astype(int))
        end = tuple(np.round(limb.end).astype(int))
        cv2.line(
            overlay,
            start,
            end,
            limb_color,
            int(max(2, limb.thickness)),
            cv2.LINE_AA,
        )
    return overlay
