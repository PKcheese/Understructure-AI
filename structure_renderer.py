"""
Stylized 3D-ish overlay construction rendered directly in 2D.

The goal is to mimic construction maquettes (boxes + cylinders) with clear
occlusion, foreshortening cues, and distinct silhouettes, similar to gesture /
structure drawing references.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from pose_utils import LMS, compute_oriented_boxes


def _bgr(r: int, g: int, b: int) -> Tuple[int, int, int]:
    return (b, g, r)


STYLE_PRESETS: Dict[str, Dict] = {
    "painterly": {
        "torso": {
            "fill": _bgr(232, 134, 90),
            "edge": _bgr(110, 60, 45),
            "highlight": _bgr(255, 205, 165),
            "alpha": 215,
            "bottom_lift": 0.28,
            "waist_taper": 0.18,
        },
        "pelvis": {
            "fill": _bgr(82, 141, 255),
            "edge": _bgr(40, 65, 140),
            "highlight": _bgr(165, 205, 255),
            "alpha": 215,
            "length_factor": 0.38,
            "flare": 0.25,
        },
        "head": {
            "fill": _bgr(250, 178, 150),
            "edge": _bgr(120, 70, 60),
            "highlight": _bgr(255, 225, 205),
            "alpha": 210,
            "width_scale": 0.55,
            "height_scale": 1.05,
        },
        "limbs": {
            "highlight_shift": 0.33,
            "categories": {
                "arm_upper": {
                    "fill": _bgr(240, 140, 110),
                    "highlight": _bgr(255, 205, 180),
                    "edge": _bgr(110, 55, 40),
                    "width_scale": 0.32,
                    "min_thickness": 16,
                    "alpha": 220,
                },
                "arm_lower": {
                    "fill": _bgr(245, 155, 120),
                    "highlight": _bgr(255, 215, 190),
                    "edge": _bgr(125, 60, 45),
                    "width_scale": 0.27,
                    "min_thickness": 14,
                    "alpha": 220,
                },
                "hand": {
                    "fill": _bgr(250, 185, 150),
                    "highlight": _bgr(255, 228, 200),
                    "edge": _bgr(140, 70, 55),
                    "width_scale": 0.22,
                    "min_thickness": 12,
                    "alpha": 215,
                },
                "leg_upper": {
                    "fill": _bgr(95, 165, 255),
                    "highlight": _bgr(170, 210, 255),
                    "edge": _bgr(35, 70, 150),
                    "width_scale": 0.36,
                    "min_thickness": 18,
                    "alpha": 220,
                },
                "leg_lower": {
                    "fill": _bgr(120, 185, 255),
                    "highlight": _bgr(190, 225, 255),
                    "edge": _bgr(40, 75, 150),
                    "width_scale": 0.30,
                    "min_thickness": 15,
                    "alpha": 220,
                },
                "foot": {
                    "fill": _bgr(140, 205, 255),
                    "highlight": _bgr(210, 235, 255),
                    "edge": _bgr(50, 85, 155),
                    "width_scale": 0.26,
                    "min_thickness": 12,
                    "alpha": 210,
                },
            },
        },
    },
    "solid": {
        "torso": {
            "fill": _bgr(210, 125, 60),
            "edge": _bgr(80, 45, 25),
            "highlight": _bgr(255, 210, 110),
            "alpha": 240,
            "bottom_lift": 0.25,
            "waist_taper": 0.15,
        },
        "pelvis": {
            "fill": _bgr(70, 110, 230),
            "edge": _bgr(35, 55, 135),
            "highlight": _bgr(150, 190, 255),
            "alpha": 235,
            "length_factor": 0.34,
            "flare": 0.20,
        },
        "head": {
            "fill": _bgr(230, 160, 140),
            "edge": _bgr(105, 55, 45),
            "highlight": _bgr(255, 225, 195),
            "alpha": 230,
            "width_scale": 0.6,
            "height_scale": 1.0,
        },
        "limbs": {
            "highlight_shift": 0.28,
            "categories": {
                "arm_upper": {
                    "fill": _bgr(220, 120, 70),
                    "highlight": _bgr(255, 190, 140),
                    "edge": _bgr(90, 45, 25),
                    "width_scale": 0.30,
                    "min_thickness": 18,
                    "alpha": 235,
                },
                "arm_lower": {
                    "fill": _bgr(230, 140, 90),
                    "highlight": _bgr(255, 210, 160),
                    "edge": _bgr(100, 50, 30),
                    "width_scale": 0.26,
                    "min_thickness": 16,
                    "alpha": 235,
                },
                "hand": {
                    "fill": _bgr(240, 170, 130),
                    "highlight": _bgr(255, 220, 180),
                    "edge": _bgr(110, 55, 35),
                    "width_scale": 0.22,
                    "min_thickness": 12,
                    "alpha": 225,
                },
                "leg_upper": {
                    "fill": _bgr(80, 135, 245),
                    "highlight": _bgr(155, 190, 255),
                    "edge": _bgr(30, 60, 150),
                    "width_scale": 0.34,
                    "min_thickness": 20,
                    "alpha": 235,
                },
                "leg_lower": {
                    "fill": _bgr(100, 160, 250),
                    "highlight": _bgr(180, 210, 255),
                    "edge": _bgr(35, 70, 150),
                    "width_scale": 0.28,
                    "min_thickness": 16,
                    "alpha": 235,
                },
                "foot": {
                    "fill": _bgr(120, 175, 255),
                    "highlight": _bgr(200, 220, 255),
                    "edge": _bgr(45, 85, 150),
                    "width_scale": 0.25,
                    "min_thickness": 12,
                    "alpha": 225,
                },
            },
        },
    },
    "sketch": {
        "torso": {
            "fill": _bgr(235, 235, 245),
            "edge": _bgr(60, 60, 80),
            "highlight": _bgr(255, 255, 255),
            "alpha": 190,
            "bottom_lift": 0.25,
            "waist_taper": 0.22,
        },
        "pelvis": {
            "fill": _bgr(220, 220, 240),
            "edge": _bgr(55, 55, 80),
            "highlight": _bgr(255, 255, 255),
            "alpha": 185,
            "length_factor": 0.36,
            "flare": 0.28,
        },
        "head": {
            "fill": _bgr(240, 240, 250),
            "edge": _bgr(70, 70, 90),
            "highlight": _bgr(255, 255, 255),
            "alpha": 190,
            "width_scale": 0.58,
            "height_scale": 1.05,
        },
        "limbs": {
            "highlight_shift": 0.35,
            "categories": {
                "arm_upper": {
                    "fill": _bgr(235, 235, 250),
                    "highlight": _bgr(255, 255, 255),
                    "edge": _bgr(70, 70, 90),
                    "width_scale": 0.30,
                    "min_thickness": 14,
                    "alpha": 200,
                },
                "arm_lower": {
                    "fill": _bgr(235, 235, 250),
                    "highlight": _bgr(255, 255, 255),
                    "edge": _bgr(70, 70, 90),
                    "width_scale": 0.25,
                    "min_thickness": 12,
                    "alpha": 200,
                },
                "hand": {
                    "fill": _bgr(242, 242, 255),
                    "highlight": _bgr(255, 255, 255),
                    "edge": _bgr(80, 80, 95),
                    "width_scale": 0.20,
                    "min_thickness": 10,
                    "alpha": 190,
                },
                "leg_upper": {
                    "fill": _bgr(220, 225, 250),
                    "highlight": _bgr(245, 245, 255),
                    "edge": _bgr(60, 65, 90),
                    "width_scale": 0.32,
                    "min_thickness": 16,
                    "alpha": 200,
                },
                "leg_lower": {
                    "fill": _bgr(225, 230, 255),
                    "highlight": _bgr(245, 245, 255),
                    "edge": _bgr(60, 65, 90),
                    "width_scale": 0.27,
                    "min_thickness": 14,
                    "alpha": 195,
                },
                "foot": {
                    "fill": _bgr(230, 235, 255),
                    "highlight": _bgr(255, 255, 255),
                    "edge": _bgr(70, 75, 95),
                    "width_scale": 0.22,
                    "min_thickness": 12,
                    "alpha": 190,
                },
            },
        },
    },
}


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


def render_structure_overlay(
    image_bgr: np.ndarray,
    k2d: np.ndarray,
    k3d: np.ndarray,
    style: str = "painterly",
) -> Tuple[np.ndarray, List]:
    """
    Build a stylized RGBA overlay representing the figure as construction primitives.

    Returns:
        overlay_rgba: np.ndarray (H, W, 4)
        boxes_3d: List of OrientedBox (pelvis, ribcage, head)
    """

    style_cfg = STYLE_PRESETS.get(style, STYLE_PRESETS["painterly"])
    height, width = image_bgr.shape[:2]

    overlay_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    overlay_alpha = np.zeros((height, width), dtype=np.uint8)

    shapes = _collect_shapes(k2d, k3d, style_cfg)
    # draw farthest shapes first (largest depth)
    shapes.sort(key=lambda item: item["depth"], reverse=True)

    for shape in shapes:
        if shape["kind"] == "box":
            _draw_box(
                overlay_rgb,
                overlay_alpha,
                shape["points"],
                style_cfg[shape["style_key"]],
            )
        elif shape["kind"] == "pelvis":
            _draw_box(
                overlay_rgb,
                overlay_alpha,
                shape["points"],
                style_cfg["pelvis"],
            )
        elif shape["kind"] == "head":
            _draw_head(
                overlay_rgb,
                overlay_alpha,
                shape["center"],
                shape["axes"],
                shape["angle"],
                style_cfg["head"],
            )
        elif shape["kind"] == "capsule":
            _draw_capsule(
                overlay_rgb,
                overlay_alpha,
                shape["p0"],
                shape["p1"],
                style_cfg["limbs"]["categories"][shape["category"]],
                style_cfg["limbs"]["highlight_shift"],
            )

    overlay_rgba = np.dstack([overlay_rgb, overlay_alpha])
    boxes_world = compute_oriented_boxes(k3d)
    return overlay_rgba, boxes_world


def composite_overlay(base_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Composite a RGBA overlay onto the base image."""
    if overlay_rgba is None or overlay_rgba.shape[2] != 4:
        return base_bgr
    overlay = overlay_rgba[..., :3].astype(np.float32)
    alpha = (overlay_rgba[..., 3:4].astype(np.float32)) / 255.0
    base = base_bgr.astype(np.float32)
    comp = overlay * alpha + base * (1.0 - alpha)
    return comp.astype(np.uint8)


def serialize_box_info(boxes: Sequence, box_names: Iterable[str] = ("pelvis", "ribcage")) -> dict:
    payload = []
    for box in boxes:
        if box.name not in box_names:
            continue
        payload.append(
            {
                "name": box.name,
                "center": box.center.tolist(),
                "axes": box.axes.tolist(),
                "size": list(box.size),
                "corners": box.corners().tolist(),
            }
        )
    return {"boxes": payload}


def _collect_shapes(k2d: np.ndarray, k3d: np.ndarray, style_cfg: Dict) -> List[Dict]:
    coords2d = {member.name: k2d[member.value, :2] for member in LMS}
    coords3d = {member.name: k3d[member.value] for member in LMS}

    shapes: List[Dict] = []

    torso_shape = _build_torso_shape(coords2d, coords3d, style_cfg["torso"])
    if torso_shape:
        shapes.append(torso_shape)

    pelvis_shape = _build_pelvis_shape(coords2d, coords3d, style_cfg["pelvis"])
    if pelvis_shape:
        shapes.append(pelvis_shape)

    head_shape = _build_head_shape(coords2d, coords3d, style_cfg["head"])
    if head_shape:
        shapes.append(head_shape)

    limb_shapes = _build_limb_shapes(coords2d, coords3d, style_cfg["limbs"]["categories"])
    shapes.extend(limb_shapes)

    return shapes


def _build_torso_shape(coords2d: Dict[str, np.ndarray], coords3d: Dict[str, np.ndarray], style: Dict) -> Dict:
    left_sh = coords2d[LMS.LEFT_SHOULDER.name]
    right_sh = coords2d[LMS.RIGHT_SHOULDER.name]
    left_hip = coords2d[LMS.LEFT_HIP.name]
    right_hip = coords2d[LMS.RIGHT_HIP.name]

    lift = style.get("bottom_lift", 0.25)
    taper = style.get("waist_taper", 0.18)

    bottom_left = left_hip + (left_sh - left_hip) * lift
    bottom_right = right_hip + (right_sh - right_hip) * lift
    bottom_center = (bottom_left + bottom_right) * 0.5
    bottom_left = bottom_center + (bottom_left - bottom_center) * (1.0 - taper)
    bottom_right = bottom_center + (bottom_right - bottom_center) * (1.0 - taper)

    points = np.array([left_sh, right_sh, bottom_right, bottom_left], dtype=np.float32)
    depth = float(
        (
            coords3d[LMS.LEFT_SHOULDER.name][2]
            + coords3d[LMS.RIGHT_SHOULDER.name][2]
            + coords3d[LMS.LEFT_HIP.name][2]
            + coords3d[LMS.RIGHT_HIP.name][2]
        )
        / 4.0
    )
    return {"kind": "box", "points": points, "depth": depth, "style_key": "torso"}


def _build_pelvis_shape(coords2d: Dict[str, np.ndarray], coords3d: Dict[str, np.ndarray], style: Dict) -> Dict:
    left_sh = coords2d[LMS.LEFT_SHOULDER.name]
    right_sh = coords2d[LMS.RIGHT_SHOULDER.name]
    left_hip = coords2d[LMS.LEFT_HIP.name]
    right_hip = coords2d[LMS.RIGHT_HIP.name]
    left_knee = coords2d[LMS.LEFT_KNEE.name]
    right_knee = coords2d[LMS.RIGHT_KNEE.name]

    lift = style.get("bottom_lift", 0.08)
    length_factor = style.get("length_factor", 0.36)
    flare = style.get("flare", 0.22)

    top_left = left_hip + (left_sh - left_hip) * lift
    top_right = right_hip + (right_sh - right_hip) * lift
    bottom_left = top_left + (left_knee - top_left) * length_factor
    bottom_right = top_right + (right_knee - top_right) * length_factor
    bottom_center = (bottom_left + bottom_right) * 0.5
    bottom_left = bottom_center + (bottom_left - bottom_center) * (1.0 + flare)
    bottom_right = bottom_center + (bottom_right - bottom_center) * (1.0 + flare)

    points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    depth = float(
        (
            coords3d[LMS.LEFT_HIP.name][2]
            + coords3d[LMS.RIGHT_HIP.name][2]
            + coords3d[LMS.LEFT_KNEE.name][2]
            + coords3d[LMS.RIGHT_KNEE.name][2]
        )
        / 4.0
    )
    return {"kind": "pelvis", "points": points, "depth": depth}


def _build_head_shape(coords2d: Dict[str, np.ndarray], coords3d: Dict[str, np.ndarray], style: Dict) -> Dict:
    nose = coords2d[LMS.NOSE.name]
    left_ear = coords2d[LMS.LEFT_EAR.name]
    right_ear = coords2d[LMS.RIGHT_EAR.name]
    left_eye = coords2d[LMS.LEFT_EYE_OUTER.name]
    right_eye = coords2d[LMS.RIGHT_EYE_OUTER.name]
    shoulder_left = coords2d[LMS.LEFT_SHOULDER.name]
    shoulder_right = coords2d[LMS.RIGHT_SHOULDER.name]

    ear_span = np.linalg.norm(right_ear - left_ear)
    if ear_span < 1e-3:
        ear_span = np.linalg.norm(right_eye - left_eye)

    if ear_span < 1e-3:
        return {}

    center = (nose * 0.4) + ((left_eye + right_eye) * 0.3) + (((shoulder_left + shoulder_right) * 0.5) * 0.3)
    axes = (
        max(12.0, ear_span * style.get("width_scale", 0.55)),
        max(18.0, ear_span * style.get("height_scale", 1.05)),
    )
    angle = np.degrees(np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0]))
    depth = float(
        (
            coords3d[LMS.NOSE.name][2]
            + coords3d[LMS.LEFT_EAR.name][2]
            + coords3d[LMS.RIGHT_EAR.name][2]
        )
        / 3.0
    )
    return {"kind": "head", "center": center, "axes": axes, "angle": angle, "depth": depth}


def _build_limb_shapes(coords2d: Dict[str, np.ndarray], coords3d: Dict[str, np.ndarray], categories: Dict) -> List[Dict]:
    shapes: List[Dict] = []
    for name, joint_a, joint_b, category in LIMB_SEGMENTS:
        p0 = coords2d[joint_a.name]
        p1 = coords2d[joint_b.name]
        length = float(np.linalg.norm(p1 - p0))
        if length < 1e-3 or category not in categories:
            continue
        style = categories[category]
        width_scale = style.get("width_scale", 0.28)
        min_thickness = style.get("min_thickness", 12)
        thickness = max(min_thickness, length * width_scale)
        depth = float((coords3d[joint_a.name][2] + coords3d[joint_b.name][2]) / 2.0)
        shapes.append(
            {
                "kind": "capsule",
                "p0": p0,
                "p1": p1,
                "radius": thickness / 2.0,
                "category": category,
                "depth": depth,
            }
        )
    return shapes


def _draw_box(canvas_rgb: np.ndarray, canvas_alpha: np.ndarray, points: np.ndarray, style: Dict) -> None:
    pts = np.round(points).astype(np.int32)
    h, w = canvas_alpha.shape
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255, lineType=cv2.LINE_AA)

    for c in range(3):
        canvas_rgb[..., c] = np.where(mask > 0, style["fill"][c], canvas_rgb[..., c])
    canvas_alpha[mask > 0] = style["alpha"]

    cv2.polylines(canvas_rgb, [pts], True, style["edge"], 2, cv2.LINE_AA)
    # diagonal guides
    cv2.line(canvas_rgb, tuple(pts[0]), tuple(pts[2]), style["highlight"], 1, cv2.LINE_AA)
    cv2.line(canvas_rgb, tuple(pts[1]), tuple(pts[3]), style["highlight"], 1, cv2.LINE_AA)


def _draw_head(canvas_rgb: np.ndarray, canvas_alpha: np.ndarray, center: np.ndarray, axes: Tuple[float, float], angle: float, style: Dict) -> None:
    h, w = canvas_alpha.shape
    center_int = tuple(np.round(center).astype(int))
    axes_int = (int(max(8, axes[0] / 2.0)), int(max(10, axes[1] / 2.0)))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center_int, axes_int, angle, 0, 360, 255, -1, cv2.LINE_AA)

    for c in range(3):
        canvas_rgb[..., c] = np.where(mask > 0, style["fill"][c], canvas_rgb[..., c])
    canvas_alpha[mask > 0] = style["alpha"]

    # highlight band
    highlight_axes = (int(axes_int[0] * 0.6), int(axes_int[1] * 0.6))
    cv2.ellipse(canvas_rgb, center_int, highlight_axes, angle - 15, -140, 40, style["highlight"], 2, cv2.LINE_AA)

    cv2.ellipse(canvas_rgb, center_int, axes_int, angle, 0, 360, style["edge"], 2, cv2.LINE_AA)
    cv2.line(
        canvas_rgb,
        (center_int[0], center_int[1] - axes_int[1]),
        (center_int[0], center_int[1] + axes_int[1]),
        style["highlight"],
        1,
        cv2.LINE_AA,
    )


def _draw_capsule(
    canvas_rgb: np.ndarray,
    canvas_alpha: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    style: Dict,
    highlight_shift: float,
) -> None:
    h, w = canvas_alpha.shape
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < 1e-3:
        return
    direction_unit = direction / length
    normal = np.array([-direction_unit[1], direction_unit[0]])
    radius = max(style.get("min_thickness", 12) / 2.0, length * style.get("width_scale", 0.3) / 2.0)
    thickness = int(max(2.0, radius * 2.0))

    p0_int = tuple(np.round(p0).astype(int))
    p1_int = tuple(np.round(p1).astype(int))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask, p0_int, p1_int, 255, thickness, cv2.LINE_AA)
    cv2.circle(mask, p0_int, thickness // 2, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, p1_int, thickness // 2, 255, -1, cv2.LINE_AA)

    for c in range(3):
        canvas_rgb[..., c] = np.where(mask > 0, style["fill"][c], canvas_rgb[..., c])
    canvas_alpha[mask > 0] = style["alpha"]

    # highlight stripe offset to one side
    highlight_offset = normal * highlight_shift * radius
    hp0 = tuple(np.round(p0 + highlight_offset).astype(int))
    hp1 = tuple(np.round(p1 + highlight_offset).astype(int))
    highlight_thick = max(1, int(thickness * 0.35))
    cv2.line(canvas_rgb, hp0, hp1, style["highlight"], highlight_thick, cv2.LINE_AA)
    cv2.circle(canvas_rgb, hp0, highlight_thick // 2, style["highlight"], -1, cv2.LINE_AA)
    cv2.circle(canvas_rgb, hp1, highlight_thick // 2, style["highlight"], -1, cv2.LINE_AA)

    # contour stroke
    edge_thick = max(1, thickness // 6)
    cv2.line(canvas_rgb, p0_int, p1_int, style["edge"], edge_thick, cv2.LINE_AA)
    cv2.circle(canvas_rgb, p0_int, edge_thick + 1, style["edge"], edge_thick, cv2.LINE_AA)
    cv2.circle(canvas_rgb, p1_int, edge_thick + 1, style["edge"], edge_thick, cv2.LINE_AA)
