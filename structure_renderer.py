"""
Render gesture structure primitives using trimesh + pyrender for shaded overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
import pyrender
import trimesh

from pose_utils import LimbSegment, OrientedBox


CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
CV_TO_GL_3 = CV_TO_GL[:3, :3]


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
    camera = pyrender.IntrinsicsCamera(
        fx=f,
        fy=f,
        cx=width / 2.0,
        cy=height / 2.0,
        znear=0.05,
        zfar=10.0,
    )
    return camera, camera_pose_gl


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
    boxes: Sequence[OrientedBox],
    limbs: Sequence[LimbSegment],
) -> np.ndarray | None:
    height, width = image_bgr.shape[:2]
    solve = _solve_camera(k2d, k3d, width, height)
    if not solve:
        return None
    camera, camera_pose = solve

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

    for box in boxes:
        mesh = _box_mesh(box, box_colors.get(box.name, (200, 200, 200)))
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    for limb in limbs:
        mesh = _cylinder_mesh(limb, limb_color)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0.0, 1.0, 2.0])
    scene.add(light, pose=light_pose)

    fill_light = pyrender.SpotLight(
        color=np.array([1.0, 0.95, 0.9]),
        intensity=3.0,
        innerConeAngle=np.pi / 12.0,
        outerConeAngle=np.pi / 6.0,
    )
    fill_pose = np.array(
        [
            [1, 0, 0, -1.0],
            [0, 0.342, -0.94, 1.8],
            [0, 0.94, 0.342, 1.5],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    scene.add(fill_light, pose=fill_pose)

    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return color


def composite_overlay(base_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    if overlay_rgba is None:
        return base_bgr
    overlay_rgb = overlay_rgba[:, :, :3][:, :, ::-1].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3:4].astype(np.float32)) / 255.0
    base = base_bgr.astype(np.float32)
    comp = overlay_rgb * alpha + base * (1.0 - alpha)
    return comp.astype(np.uint8)
