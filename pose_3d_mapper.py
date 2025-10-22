"""
Generate a JSON dump of MediaPipe Pose 3D coordinates and visualize them on the image.

Steps:
- load an input photo
- run MediaPipe Pose (static image mode) via pose_utils.run_pose_estimation
- collect pixel-space, normalized, and world (metric) coordinates per landmark
- save a JSON file for downstream processing
- render an overlay that annotates each landmark with its 3D coordinate snippet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from pose_utils import run_pose_estimation, LMS


# Canonical MediaPipe skeleton connections expressed as ordered landmark pairs.
# Keeping this local (instead of relying on the unordered frozenset exposed by
# mediapipe) prevents jittering/“random” bone renderings across frames.
POSE_CONNECTIONS_ORDERED: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 4),
    (1, 2),
    (2, 3),
    # Face loop
    (3, 7),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    # Torso
    (11, 12),
    (11, 13),
    (11, 23),
    (12, 14),
    (12, 24),
    (13, 15),
    (14, 16),
    # Left arm
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    # Right arm
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    # Pelvis
    (23, 24),
    # Left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    # Right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
)

POSE_LANDMARK_CONNECTIONS: Tuple[Tuple[LMS, LMS], ...] = tuple(
    (LMS(a), LMS(b)) for a, b in POSE_CONNECTIONS_ORDERED
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Pose 3D coordinates and visualize them."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input photo path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pose3d_outputs"),
        help="Directory to save JSON, blank projection, and rendering.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.2,
        help="Visibility threshold to annotate a landmark on the image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr, landmarks_2d, landmarks_3d = run_pose_estimation(args.image)
    height, width = image_bgr.shape[:2]

    payload = {}
    overlay = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for member in LMS:
        idx = member.value
        x_pix, y_pix, visibility = landmarks_2d[idx]
        norm_x = x_pix / width
        norm_y = y_pix / height
        world_xyz = landmarks_3d[idx]

        payload[member.name] = {
            "pixel": [float(x_pix), float(y_pix)],
            "normalized": [float(norm_x), float(norm_y)],
            "world": [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])],
            "visibility": float(visibility),
        }

        if visibility >= args.min_visibility:
            center = (int(round(x_pix)), int(round(y_pix)))
            cv2.circle(overlay, center, 5, (0, 80, 255), -1, cv2.LINE_AA)
            text = f"{member.name.lower()} z={world_xyz[2]:.3f}"
            cv2.putText(
                overlay,
                text,
                (center[0] + 6, center[1] - 6),
                font,
                0.35,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    # Create blank-space render of the 3D coordinates (orthographic projection).
    blank_top, blank_front = _render_blank_views(landmarks_3d)

    json_path = args.output_dir / "pose_3d_coordinates.json"
    json_path.write_text(json.dumps(payload, indent=2))

    overlay_path = args.output_dir / "pose_3d_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    blank_top_path = args.output_dir / "pose3d_blank_top.png"
    cv2.imwrite(str(blank_top_path), blank_top)

    blank_front_path = args.output_dir / "pose3d_blank_front.png"
    cv2.imwrite(str(blank_front_path), blank_front)

    print(f"Saved JSON with 3D coordinates to {json_path}")
    print(f"Saved annotated overlay to {overlay_path}")
    print(f"Saved blank-space top projection to {blank_top_path}")
    print(f"Saved blank-space front projection to {blank_front_path}")


def _render_blank_views(
    landmarks_3d: np.ndarray,
    canvas_size: Tuple[int, int] = (720, 720),
) -> Tuple[np.ndarray, np.ndarray]:
    """Render two orthographic projections of the 3D points on blank backgrounds."""

    joints = {member.value: landmarks_3d[member.value] for member in LMS}

    blank_top = _draw_projection(joints, POSE_CONNECTIONS_ORDERED, canvas_size, project="top")
    blank_front = _draw_projection(joints, POSE_CONNECTIONS_ORDERED, canvas_size, project="front")
    return blank_top, blank_front


def _draw_projection(
    joints: Dict[int, np.ndarray],
    bones: Sequence[Tuple[int, int]],
    canvas_size: Tuple[int, int],
    project: str,
) -> np.ndarray:
    canvas = np.zeros((*canvas_size, 3), dtype=np.uint8)
    h, w = canvas_size

    ordered_indices = [member.value for member in LMS]
    projected_points: list[np.ndarray] = []
    for idx in ordered_indices:
        coord = joints[idx]
        if project == "top":
            x, y = coord[0], -coord[2]
        elif project == "front":
            x, y = coord[0], coord[1]
        else:
            raise ValueError("Unknown projection type")
        projected_points.append(np.array([x, y], dtype=np.float32))

    arr = np.array(projected_points, dtype=np.float32)
    if arr.size == 0:
        return canvas
    mean = arr.mean(axis=0)
    arr -= mean
    max_extent = np.abs(arr).max() or 1.0
    scale = (min(w, h) * 0.35) / max_extent
    arr *= scale
    arr[:, 0] += w / 2
    arr[:, 1] += h / 2

    joint_pixels = {idx: coords for idx, coords in zip(ordered_indices, arr)}

    # draw bones
    for a, b in bones:
        pa = joint_pixels.get(a)
        pb = joint_pixels.get(b)
        if pa is None or pb is None:
            continue
        cv2.line(
            canvas,
            tuple(np.round(pa).astype(int)),
            tuple(np.round(pb).astype(int)),
            (30, 30, 220),
            2,
            cv2.LINE_AA,
        )

    # draw joints
    for landmark, pt in joint_pixels.items():
        center = tuple(np.round(pt).astype(int))
        cv2.circle(canvas, center, 5, (0, 80, 255), -1, cv2.LINE_AA)

    return canvas


if __name__ == "__main__":
    main()
