"""
Render a sculptural construction drawing from MediaPipe pose landmarks.

The script interprets 3D landmark coordinates as simple primitives:
- cylinders for limbs with tonal bands to suggest volume
- tapered boxes for ribcage and pelvis
- an egg-shaped head block

Depth cues (draw order, atmospheric lightening, drop shadow) help separate
foreground from background, producing a clean structural sketch similar to
figure-construction reference sheets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import json
import os

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import numpy as np

from pose_utils import (
    LMS,
    compute_oriented_boxes_2d,
    run_pose_estimation,
)


# Limb definitions: (name, start_joint, end_joint, visual category)
LIMB_SEGMENTS: Sequence[Tuple[str, LMS, LMS, str]] = [
    ("left_upper_arm", LMS.LEFT_SHOULDER, LMS.LEFT_ELBOW, "arm_upper"),
    ("left_forearm", LMS.LEFT_ELBOW, LMS.LEFT_WRIST, "arm_lower"),
    ("left_hand", LMS.LEFT_WRIST, LMS.LEFT_INDEX, "hand"),
    ("right_upper_arm", LMS.RIGHT_SHOULDER, LMS.RIGHT_ELBOW, "arm_upper"),
    ("right_forearm", LMS.RIGHT_ELBOW, LMS.RIGHT_WRIST, "arm_lower"),
    ("right_hand", LMS.RIGHT_WRIST, LMS.RIGHT_INDEX, "hand"),
    ("left_thigh", LMS.LEFT_HIP, LMS.LEFT_KNEE, "leg_upper"),
    ("left_calf", LMS.LEFT_KNEE, LMS.LEFT_ANKLE, "leg_lower"),
    ("left_foot", LMS.LEFT_ANKLE, LMS.LEFT_FOOT_INDEX, "foot"),
    ("right_thigh", LMS.RIGHT_HIP, LMS.RIGHT_KNEE, "leg_upper"),
    ("right_calf", LMS.RIGHT_KNEE, LMS.RIGHT_ANKLE, "leg_lower"),
    ("right_foot", LMS.RIGHT_ANKLE, LMS.RIGHT_FOOT_INDEX, "foot"),
]


# Width heuristics tuned for a sketch aesthetic. Values represent the radius as a
# fraction of the projected limb length, plus a minimum radius to avoid collapse.
LIMB_STYLE: Dict[str, Dict[str, float]] = {
    "arm_upper": {"width_scale": 0.27, "min_radius": 12.0},
    "arm_lower": {"width_scale": 0.22, "min_radius": 10.0},
    "hand": {"width_scale": 0.18, "min_radius": 8.0},
    "leg_upper": {"width_scale": 0.30, "min_radius": 14.0},
    "leg_lower": {"width_scale": 0.24, "min_radius": 12.0},
    "foot": {"width_scale": 0.20, "min_radius": 9.0},
}


BOX_DEPTH_MARKERS: Dict[str, Sequence[LMS]] = {
    "pelvis": (LMS.LEFT_HIP, LMS.RIGHT_HIP),
    "ribcage": (LMS.LEFT_SHOULDER, LMS.RIGHT_SHOULDER),
    "head": (LMS.NOSE, LMS.LEFT_EAR, LMS.RIGHT_EAR),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce a sculptural figure drawing from MediaPipe pose landmarks."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input photo path (used for sizing).")
    parser.add_argument(
        "--pose-json",
        type=Path,
        help="Optional MediaPipe pose JSON (as produced by pose_3d_mapper.py) to reuse existing landmarks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sculptural_drawing.png"),
        help="Output path for the rendered drawing.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.35,
        help="Visibility threshold for using a landmark in the construction.",
    )
    parser.add_argument(
        "--shadow-strength",
        type=float,
        default=55.0,
        help="Maximum darkness applied to the drop shadow (0-100 recommended).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pose_json:
        image_bgr = cv2.imread(str(args.image))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image at {args.image}")
        k2d, k3d = _load_pose_from_json(args.pose_json)
    else:
        image_bgr, k2d, k3d = run_pose_estimation(args.image)
    height, width = image_bgr.shape[:2]

    figure_layer = np.zeros((height, width, 3), dtype=np.uint8)
    figure_mask = np.zeros((height, width), dtype=np.uint8)

    depth_min = float(np.min(k3d[:, 2]))
    depth_max = float(np.max(k3d[:, 2]))
    depth_range = max(depth_max - depth_min, 1e-5)

    shapes = _gather_shapes(k2d, k3d, args.min_visibility)
    shapes.sort(key=lambda item: item["depth"], reverse=True)  # farthest first

    for shape in shapes:
        tones = _depth_tones(shape["depth"], depth_min, depth_range)
        if shape["kind"] == "cylinder":
            _draw_cylinder(
                figure_layer,
                figure_mask,
                shape["start"],
                shape["end"],
                shape["radius"],
                tones,
            )
        elif shape["kind"] == "box":
            _draw_box(
                figure_layer,
                figure_mask,
                shape["corners"],
                tones,
            )
        elif shape["kind"] == "head":
            _draw_head(
                figure_layer,
                figure_mask,
                shape["center"],
                shape["axes"],
                shape["size"],
                tones,
            )

    paper = _make_paper_texture(height, width)
    paper = _apply_drop_shadow(paper, figure_mask, darkness=args.shadow_strength)
    final = _composite(paper, figure_layer, figure_mask)

    cv2.imwrite(str(args.output), final)
    print(f"Sculptural drawing saved to {args.output}")


def _gather_shapes(
    k2d: np.ndarray,
    k3d: np.ndarray,
    min_visibility: float,
) -> List[Dict]:
    shapes: List[Dict] = []

    # Oriented boxes (pelvis, ribcage, head shell)
    boxes_2d = compute_oriented_boxes_2d(k2d)
    for box in boxes_2d:
        markers = BOX_DEPTH_MARKERS.get(box.name, ())
        if not markers:
            continue
        depth = float(np.mean([k3d[m.value, 2] for m in markers]))
        corners = np.array(box.corners(), dtype=np.float32)
        if box.name == "head":
            shapes.append(
                {
                    "kind": "head",
                    "center": np.array(box.center, dtype=np.float32),
                    "axes": np.array(box.axes, dtype=np.float32),
                    "size": np.array(box.size, dtype=np.float32),
                    "depth": depth,
                }
            )
        else:
            shapes.append(
                {
                    "kind": "box",
                    "name": box.name,
                    "corners": corners,
                    "depth": depth,
                }
            )

    # Cylindrical limbs
    for name, joint_a, joint_b, category in LIMB_SEGMENTS:
        vis_a = k2d[joint_a.value, 2]
        vis_b = k2d[joint_b.value, 2]
        if vis_a < min_visibility or vis_b < min_visibility:
            continue

        start = k2d[joint_a.value, :2].astype(np.float32)
        end = k2d[joint_b.value, :2].astype(np.float32)
        length = float(np.linalg.norm(end - start))
        if length < 4.0:
            continue

        style = LIMB_STYLE[category]
        radius = max(length * style["width_scale"], style["min_radius"])
        depth = float((k3d[joint_a.value, 2] + k3d[joint_b.value, 2]) / 2.0)

        shapes.append(
            {
                "kind": "cylinder",
                "name": name,
                "start": start,
                "end": end,
                "radius": radius,
                "depth": depth,
            }
        )

    return shapes


def _load_pose_from_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text())
    count = len(LMS)
    k2d = np.zeros((count, 3), dtype=np.float32)
    k3d = np.zeros((count, 3), dtype=np.float32)
    for member in LMS:
        record = payload.get(member.name)
        if record is None:
            raise KeyError(f"Missing landmark {member.name} in {path}")
        k2d[member.value, :2] = record["pixel"]
        k2d[member.value, 2] = record.get("visibility", 1.0)
        k3d[member.value] = record["world"]
    return k2d, k3d


def _depth_tones(depth: float, depth_min: float, depth_range: float) -> Dict[str, Tuple[int, int, int]]:
    t = np.clip((depth - depth_min) / depth_range, 0.0, 1.0)
    base_val = int(np.clip(210 - (1.0 - t) * 70, 120, 220))
    highlight_val = int(np.clip(base_val + 35, 150, 245))
    shadow_val = int(np.clip(base_val - 45, 40, 200))
    outline_val = int(np.clip(base_val - 60, 30, 120))
    return {
        "base": (base_val, base_val, base_val),
        "highlight": (highlight_val, highlight_val, highlight_val),
        "shadow": (shadow_val, shadow_val, shadow_val),
        "outline": (outline_val, outline_val, outline_val),
    }


def _draw_cylinder(
    layer: np.ndarray,
    mask: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    tones: Dict[str, Tuple[int, int, int]],
) -> None:
    start_pt = np.asarray(start, dtype=np.float32)
    end_pt = np.asarray(end, dtype=np.float32)
    axis = end_pt - start_pt
    length = float(np.linalg.norm(axis))
    if length < 1e-3:
        return

    direction = axis / length
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    offset = normal * radius

    quad = np.array(
        [
            start_pt + offset,
            end_pt + offset,
            end_pt - offset,
            start_pt - offset,
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(layer, quad, tones["base"], cv2.LINE_AA)
    cv2.fillConvexPoly(mask, quad, 255, cv2.LINE_AA)

    highlight_offset = normal * radius * 0.55
    shadow_offset = -normal * radius * 0.6

    highlight_poly = np.array(
        [
            start_pt + highlight_offset * 0.4,
            end_pt + highlight_offset * 0.4,
            end_pt + highlight_offset * 0.9,
            start_pt + highlight_offset * 0.9,
        ],
        dtype=np.int32,
    )
    shadow_poly = np.array(
        [
            start_pt + shadow_offset * 0.2,
            end_pt + shadow_offset * 0.2,
            end_pt + shadow_offset * 0.9,
            start_pt + shadow_offset * 0.9,
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(layer, highlight_poly, tones["highlight"], cv2.LINE_AA)
    cv2.fillConvexPoly(layer, shadow_poly, tones["shadow"], cv2.LINE_AA)

    for center in (start_pt, end_pt):
        center_int = tuple(np.round(center).astype(int))
        rad = int(max(radius, 1.0))
        cv2.circle(layer, center_int, rad, tones["base"], -1, cv2.LINE_AA)
        cv2.circle(mask, center_int, rad, 255, cv2.LINE_AA)
        cv2.circle(layer, center_int, rad, tones["outline"], 2, cv2.LINE_AA)

    cv2.polylines(layer, [quad], True, tones["outline"], 3, cv2.LINE_AA)


def _draw_box(
    layer: np.ndarray,
    mask: np.ndarray,
    corners: np.ndarray,
    tones: Dict[str, Tuple[int, int, int]],
) -> None:
    poly = np.round(corners).astype(np.int32)
    cv2.fillConvexPoly(layer, poly, tones["base"], cv2.LINE_AA)
    cv2.fillConvexPoly(mask, poly, 255, cv2.LINE_AA)

    # Emphasize two light-facing edges and two shadow-side edges.
    highlight_edges = [(poly[0], poly[1]), (poly[0], poly[2])]
    shadow_edges = [(poly[3], poly[1]), (poly[3], poly[2])]
    for a, b in highlight_edges:
        cv2.line(layer, tuple(a), tuple(b), tones["highlight"], 4, cv2.LINE_AA)
    for a, b in shadow_edges:
        cv2.line(layer, tuple(a), tuple(b), tones["shadow"], 6, cv2.LINE_AA)

    cv2.polylines(layer, [poly], True, tones["outline"], 3, cv2.LINE_AA)


def _draw_head(
    layer: np.ndarray,
    mask: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    size: np.ndarray,
    tones: Dict[str, Tuple[int, int, int]],
) -> None:
    center_int = tuple(np.round(center).astype(int))
    axis_lengths = (int(max(size[0] * 0.5, 6.0)), int(max(size[1] * 0.45, 8.0)))
    angle = float(np.degrees(np.arctan2(axes[0, 1], axes[0, 0])))

    cv2.ellipse(layer, center_int, axis_lengths, angle, 0, 360, tones["base"], -1, cv2.LINE_AA)
    cv2.ellipse(mask, center_int, axis_lengths, angle, 0, 360, 255, -1, cv2.LINE_AA)

    highlight_axes = (int(axis_lengths[0] * 0.7), int(axis_lengths[1] * 0.7))
    cv2.ellipse(
        layer,
        center_int,
        highlight_axes,
        angle,
        -30,
        40,
        tones["highlight"],
        3,
        cv2.LINE_AA,
    )

    cv2.ellipse(
        layer,
        center_int,
        axis_lengths,
        angle,
        140,
        230,
        tones["shadow"],
        4,
        cv2.LINE_AA,
    )

    cv2.ellipse(layer, center_int, axis_lengths, angle, 0, 360, tones["outline"], 3, cv2.LINE_AA)

    # Simple construction lines for the face plane.
    cv2.line(
        layer,
        _rotate_point(center, axes[:, 0] * -axis_lengths[0] * 0.8),
        _rotate_point(center, axes[:, 0] * axis_lengths[0] * 0.8),
        tones["outline"],
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        layer,
        _rotate_point(center, axes[:, 1] * -axis_lengths[1] * 0.6),
        _rotate_point(center, axes[:, 1] * axis_lengths[1] * 0.8),
        tones["outline"],
        2,
        cv2.LINE_AA,
    )


def _rotate_point(center: np.ndarray, offset: np.ndarray) -> Tuple[int, int]:
    pt = center + offset
    return tuple(np.round(pt).astype(int))


def _make_paper_texture(height: int, width: int) -> np.ndarray:
    paper = np.full((height, width, 3), 245, dtype=np.uint8)

    vertical = np.linspace(18, -10, height, dtype=np.float32).reshape(height, 1, 1)
    horizontal = np.linspace(12, -6, width, dtype=np.float32).reshape(1, width, 1)
    paper = np.clip(paper.astype(np.float32) - vertical - horizontal, 180, 255).astype(np.uint8)

    cv2.rectangle(paper, (24, 24), (width - 25, height - 25), (200, 200, 200), 2, cv2.LINE_AA)
    return paper


def _apply_drop_shadow(
    base: np.ndarray,
    mask: np.ndarray,
    offset: Tuple[int, int] = (20, 28),
    darkness: float = 55.0,
    blur_sigma: float = 18.0,
) -> np.ndarray:
    if darkness <= 0:
        return base

    shadow = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    matrix = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    shifted = cv2.warpAffine(shadow, matrix, (base.shape[1], base.shape[0]), borderValue=0)

    shadow_strength = (shifted.astype(np.float32) / 255.0) * float(darkness)
    shadow_rgb = np.stack([shadow_strength] * 3, axis=2)

    shaded = base.astype(np.float32) - shadow_rgb
    np.clip(shaded, 0, 255, out=shaded)
    return shaded.astype(np.uint8)


def _composite(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(overlay.astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha), 0, 255).astype(
        np.uint8
    )


if __name__ == "__main__":
    main()
