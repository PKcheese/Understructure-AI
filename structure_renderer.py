"""
Render gesture structure primitives using trimesh + pyrender for shaded overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyrender
import trimesh

from pose_utils import LimbSegment, OrientedBox


CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
CV_TO_GL_3 = CV_TO_GL[:3, :3]


@dataclass
class AlignedStructure:
    landmarks: np.ndarray
    boxes: Sequence[OrientedBox]
    limbs: Sequence[LimbSegment]
    rotation: np.ndarray
    scale: float
    translation_xy: np.ndarray
    origin_3d: np.ndarray


def _solve_camera(
    k2d: np.ndarray, k3d: np.ndarray, width: int, height: int
) -> tuple[pyrender.camera.IntrinsicsCamera, np.ndarray] | None:
    valid = k2d[:, 2] > 0.4
    if valid.sum() < 6:
        return None

    object_points = k3d[valid].astype(np.float32)
    image_points = k2d[valid, :2].astype(np.float32)

    f = float(max(width, height))
    camera_matrix = np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return None
    R, _ = cv2.Rodrigues(rvec)
    camera_pose_cv = np.eye(4)
    camera_pose_cv[:3, :3] = R.T
    camera_pose_cv[:3, 3] = (-R.T @ tvec).reshape(3)

    camera_pose_gl = CV_TO_GL @ camera_pose_cv
    yfov = 2.0 * np.arctan((height / 2.0) / f)
    camera = pyrender.PerspectiveCamera(
        yfov=float(yfov),
        aspectRatio=width / height,
        znear=0.05,
        zfar=10.0,
    )
    return camera, camera_pose_gl


def _fallback_camera(
    k3d: np.ndarray, width: int, height: int
) -> tuple[pyrender.camera.PerspectiveCamera, np.ndarray]:
    bbox_min = k3d.min(axis=0)
    bbox_max = k3d.max(axis=0)
    center = (bbox_max + bbox_min) * 0.5
    extent = np.max(bbox_max - bbox_min)
    if extent < 1e-3:
        extent = 0.5
    distance = extent * 3.0
    eye = center + np.array([0.0, 0.0, distance])
    target = center
    up = np.array([0.0, -1.0, 0.0])
    camera_pose_cv = _look_at_pose(eye, target, up)
    camera_pose_gl = CV_TO_GL @ camera_pose_cv
    yfov = np.deg2rad(55.0)
    camera = pyrender.PerspectiveCamera(
        yfov=float(yfov),
        aspectRatio=width / height,
        znear=0.05,
        zfar=10.0,
    )
    return camera, camera_pose_gl


def align_landmarks_orthographic(
    k2d: np.ndarray,
    k3d: np.ndarray,
    visibility_threshold: float = 0.4,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    valid = k2d[:, 2] > visibility_threshold
    if valid.sum() < 6:
        valid = k2d[:, 2] > 0.0
    X = k3d[valid]
    Y = k2d[valid, :2]
    x_mean = X.mean(axis=0)
    y_mean = Y.mean(axis=0)
    Xc = X - x_mean
    Yc = Y - y_mean
    Yc3 = np.concatenate([Yc, np.zeros((Yc.shape[0], 1))], axis=1)
    H = Xc.T @ Yc3
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    scale = np.trace(np.diag(S)) / np.sum(Xc ** 2)
    aligned = scale * ((k3d - x_mean) @ R)
    aligned[:, :2] += y_mean
    return aligned, scale, R, y_mean, x_mean


def _box_mesh(box: OrientedBox, color: Sequence[int]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=box.size)
    transform = np.eye(4)
    transform[:3, :3] = box.axes
    transform[:3, 3] = box.center
    mesh.apply_transform(transform)
    mesh.apply_transform(CV_TO_GL)
    color_rgba = np.array([*color, 255], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(color_rgba, (mesh.vertices.shape[0], 1))
    return mesh


def _cylinder_mesh(limb: LimbSegment, color: Sequence[int]) -> trimesh.Trimesh:
    start = limb.start
    end = limb.end
    direction = end - start
    length = np.linalg.norm(direction)
    color_rgba = np.array([*color, 255], dtype=np.uint8)
    if length < 1e-6:
        mesh = trimesh.creation.icosphere(radius=limb.radius * 0.5)
        mesh.apply_translation(start)
        mesh.apply_transform(CV_TO_GL)
        mesh.visual.vertex_colors = np.tile(color_rgba, (mesh.vertices.shape[0], 1))
        return mesh
    mesh = trimesh.creation.cylinder(
        radius=limb.radius,
        height=length,
        sections=24,
    )
    direction_unit = direction / length
    align = trimesh.geometry.align_vectors(np.array([0, 0, 1.0]), direction_unit)
    mesh.apply_transform(align)
    mesh.apply_translation((start + end) / 2.0)
    mesh.apply_transform(CV_TO_GL)
    mesh.visual.vertex_colors = np.tile(color_rgba, (mesh.vertices.shape[0], 1))
    return mesh


def render_structure_overlay(
    image_bgr: np.ndarray,
    k2d: np.ndarray,
    k3d: np.ndarray,
    boxes: Optional[Sequence[OrientedBox]] = None,
    limbs: Optional[Sequence[LimbSegment]] = None,
) -> tuple[Optional[np.ndarray], AlignedStructure]:
    height, width = image_bgr.shape[:2]
    aligned_landmarks, scale, rotation, translation_xy, origin_3d = align_landmarks_orthographic(
        k2d, k3d
    )
    aligned_landmarks[:, 1] = height - aligned_landmarks[:, 1]

    if boxes is None or limbs is None:
        from pose_utils import compute_oriented_boxes, compute_limb_segments

        boxes = compute_oriented_boxes(aligned_landmarks)
        limbs = compute_limb_segments(aligned_landmarks)

    offset = np.array([width / 2.0, height / 2.0, 0.0])
    shifted_boxes = [_shift_box(box, offset) for box in boxes]
    shifted_limbs = [_shift_limb(limb, offset) for limb in limbs]

    attempts: list[tuple[pyrender.camera.Camera, np.ndarray]] = []
    solve = _solve_camera(k2d, k3d, width, height)
    if solve:
        attempts.append(solve)
    attempts.append(_fallback_camera(aligned_landmarks, width, height))

    ortho_cam = pyrender.OrthographicCamera(
        xmag=width / 2.0,
        ymag=height / 2.0,
        znear=0.05,
        zfar=5000.0,
    )
    ortho_pose_cv = _look_at_pose(
        np.array([0.0, 0.0, 1500.0]),
        np.zeros(3),
        np.array([0.0, 1.0, 0.0]),
    )
    ortho_pose_gl = CV_TO_GL @ ortho_pose_cv
    attempts.append((ortho_cam, ortho_pose_gl))

    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        ambient_light=np.array([0.1, 0.1, 0.1, 1.0]),
    )

    box_colors = {
        "pelvis": (80, 150, 255),
        "ribcage": (255, 160, 60),
        "head": (220, 120, 255),
    }
    limb_color = (120, 255, 120)

    for box in shifted_boxes:
        mesh = _box_mesh(box, box_colors.get(box.name, (200, 200, 200)))
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    for limb in shifted_limbs:
        mesh = _cylinder_mesh(limb, limb_color)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    def add_lights(scn: pyrender.Scene) -> None:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0.0, 1.0, 2.5])
        scn.add(light, pose=light_pose)

        rim_light = pyrender.DirectionalLight(color=np.array([0.6, 0.6, 1.0]), intensity=1.8)
        rim_pose = np.array(
            [
                [0.6, 0.0, 0.8, -1.5],
                [0.0, 1.0, 0.0, 1.2],
                [-0.8, 0.0, 0.6, 1.8],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        scn.add(rim_light, pose=rim_pose)
        fill = pyrender.PointLight(color=np.array([1.0, 0.9, 0.8]), intensity=15.0)
        fill_pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.4],
                [0.0, 1.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.8],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        scn.add(fill, pose=fill_pose)

    add_lights(scene)

    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        for camera, camera_pose in attempts:
            node = scene.add(camera, pose=camera_pose)
            color, _ = renderer.render(
                scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT
            )
            scene.remove_node(node)
            if color[..., 3].max() > 0:
                info = AlignedStructure(
                    landmarks=aligned_landmarks,
                    boxes=boxes,
                    limbs=limbs,
                    rotation=rotation,
                    scale=scale,
                    translation_xy=translation_xy,
                    origin_3d=origin_3d,
                )
                return color, info
    finally:
        renderer.delete()
    info = AlignedStructure(
        landmarks=aligned_landmarks,
        boxes=boxes,
        limbs=limbs,
        rotation=rotation,
        scale=scale,
        translation_xy=translation_xy,
        origin_3d=origin_3d,
    )
    return None, info


def composite_overlay(base_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    if overlay_rgba is None:
        return base_bgr
    overlay_rgb = overlay_rgba[:, :, :3][:, :, ::-1].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3:4].astype(np.float32)) / 255.0
    alpha = np.clip(alpha * 0.6, 0.0, 1.0)
    base = base_bgr.astype(np.float32)
    comp = overlay_rgb * alpha + base * (1.0 - alpha)
    return comp.astype(np.uint8)


def _look_at_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward = forward / norm
    up = up / (np.linalg.norm(up) + 1e-8)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    up_corrected = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[0, :3] = right
    pose[1, :3] = up_corrected
    pose[2, :3] = -forward
    pose[:3, 3] = eye
    return pose


def _shift_box(box: OrientedBox, offset: np.ndarray) -> OrientedBox:
    return OrientedBox(
        name=box.name,
        center=box.center - offset,
        axes=box.axes,
        size=box.size,
    )


def _shift_limb(limb: LimbSegment, offset: np.ndarray) -> LimbSegment:
    return LimbSegment(
        name=limb.name,
        start=limb.start - offset,
        end=limb.end - offset,
        radius=limb.radius,
    )


def serialize_box_info(
    struct_info: AlignedStructure, box_names: Iterable[str] = ("pelvis", "ribcage")
) -> dict:
    boxes_payload = []
    for box in struct_info.boxes:
        if box.name not in box_names:
            continue
        boxes_payload.append(
            {
                "name": box.name,
                "center": box.center.tolist(),
                "axes": box.axes.tolist(),
                "size": list(box.size),
                "corners": box.corners().tolist(),
            }
        )
    return {
        "boxes": boxes_payload,
        "transform": {
            "scale": struct_info.scale,
            "rotation": struct_info.rotation.tolist(),
            "translation_xy": struct_info.translation_xy.tolist(),
            "origin_3d": struct_info.origin_3d.tolist(),
        },
    }
