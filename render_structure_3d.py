"""
Build a 3D maquette (boxes + cylinders) from MediaPipe pose landmarks,
render it with real depth, and export both a screenshot and a turntable-ready
mesh file.

The pipeline:
1. Gather pose landmarks (either by running MediaPipe or reusing a saved JSON
   produced by pose_3d_mapper.py).
2. Convert torso landmarks into oriented boxes and limbs into cylinders.
3. Export the assembled mesh as a GLB/OBJ for downstream tweaking.
4. Render the mesh with pyrender's offscreen renderer to produce a shaded PNG.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import numpy as np
try:
    import pyrender  # type: ignore
except Exception:
    os.environ["PYOPENGL_PLATFORM"] = "osx"
    import pyrender  # type: ignore

import trimesh

from pose_utils import (
    LMS,
    compute_limb_segments,
    compute_oriented_boxes,
    run_pose_estimation,
)


BOX_COLOR_LOOKUP: Dict[str, Tuple[int, int, int, int]] = {
    "pelvis": (205, 150, 90, 255),
    "ribcage": (170, 190, 235, 255),
    "head": (215, 185, 170, 255),
}

CYLINDER_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "arm": (230, 160, 120, 255),
    "leg": (120, 170, 240, 255),
}


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


LIGHT_DIRECTION = _normalize(np.array([-1.0, 0.35, 0.6], dtype=np.float32))

LIMB_JOINT_MAP: Dict[str, Tuple[str, str]] = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "left_hand": ("left_wrist", "left_index"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "right_hand": ("right_wrist", "right_index"),
    "left_thigh": ("left_hip", "left_knee"),
    "left_calf": ("left_knee", "left_ankle"),
    "left_foot": ("left_ankle", "left_foot"),
    "right_thigh": ("right_hip", "right_knee"),
    "right_calf": ("right_knee", "right_ankle"),
    "right_foot": ("right_ankle", "right_foot"),
}


def _apply_transform_to_metadata(mesh: trimesh.Trimesh, transform: np.ndarray) -> None:
    metadata = getattr(mesh, "metadata", None)
    if not metadata:
        return

    if metadata.get("type") == "box":
        center = np.array(metadata.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
        axes = np.array(metadata.get("axes", np.eye(3)), dtype=np.float32)
        rot = transform[:3, :3]
        trans = transform[:3, 3]
        center = rot @ center + trans
        axes = rot @ axes
        metadata["center"] = center.tolist()
        metadata["axes"] = axes.tolist()
    elif metadata.get("type") == "limb_segment":
        rot = transform[:3, :3]
        trans = transform[:3, 3]
        if "start" in metadata:
            start = np.array(metadata["start"], dtype=np.float32)
            metadata["start"] = (rot @ start + trans).tolist()
        if "end" in metadata:
            end = np.array(metadata["end"], dtype=np.float32)
            metadata["end"] = (rot @ end + trans).tolist()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 3D construction rig and export a mesh + screenshot."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input photo path.")
    parser.add_argument(
        "--pose-json",
        type=Path,
        help="Optional pose JSON (from pose_3d_mapper.py) to avoid re-running MediaPipe.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("structure_overlay_3d.png"),
        help="Output PNG rendered from the 3D scene.",
    )
    parser.add_argument(
        "--output-mesh",
        type=Path,
        default=Path("structure_maquette.glb"),
        help="Mesh export path (.glb, .obj, .ply, etc.).",
    )
    parser.add_argument(
        "--output-mesh-modified",
        type=Path,
        help="Optional second mesh path with aesthetic tweaks (red joints, outlined blocks).",
    )
    parser.add_argument(
        "--outline-mode",
        choices=("local", "filtered"),
        default="local",
        help="Method for drawing box outlines when building the modified mesh.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=960,
        help="Width/height (pixels) of the rendered screenshot.",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=35.0,
        help="Camera elevation angle in degrees (positive pitches downward).",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=25.0,
        help="Camera azimuth in degrees around the vertical axis.",
    )
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=3.0,
        help="Multiplier on the bounding radius for camera distance.",
    )
    parser.add_argument(
        "--sections",
        type=int,
        default=24,
        help="Number of radial segments for limb cylinders (higher = smoother).",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.25,
        help="Visibility threshold for including bones/boxes.",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="#f2f2f3",
        help="Hex color for the render background.",
    )
    parser.add_argument(
        "--match-image-view",
        action="store_true",
        help="Rotate the rig so the render matches the photo's viewing angle.",
    )
    parser.add_argument(
        "--render-modes",
        nargs="+",
        default=["matplotlib"],
        choices=["matplotlib", "orthographic", "wireframe", "toon", "hatch"],
        help="Rendering styles to generate. Each mode produces its own PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pose_json:
        k2d, k3d = _load_pose_from_json(args.pose_json)
    else:
        _, k2d, k3d = run_pose_estimation(args.image)

    boxes = compute_oriented_boxes(k3d)
    limbs = compute_limb_segments(k3d)

    alignment = None
    if args.match_image_view:
        alignment = _compute_view_alignment(k2d, k3d, args.min_visibility)

    meshes = _build_meshes(boxes, limbs, args.sections, args.min_visibility)

    if not meshes:
        raise RuntimeError("No meshes were generated; check visibility threshold or landmarks.")

    if alignment is not None:
        for mesh in meshes:
            mesh.apply_transform(alignment)
            _apply_transform_to_metadata(mesh, alignment)

    combined = trimesh.util.concatenate(meshes)
    combined.visual.vertex_colors = np.vstack([mesh.visual.vertex_colors for mesh in meshes])
    combined.export(str(args.output_mesh))
    print(f"Saved mesh to {args.output_mesh}")

    if args.output_mesh_modified:
        modified_meshes = _clone_meshes_with_modifications(meshes, outline_mode=args.outline_mode)
        combined_mod = trimesh.util.concatenate(modified_meshes)
        combined_mod.visual.vertex_colors = np.vstack([mesh.visual.vertex_colors for mesh in modified_meshes])
        combined_mod.export(str(args.output_mesh_modified))
        print(f"Saved modified mesh to {args.output_mesh_modified}")

    bg_rgba = _hex_to_rgba(args.background)
    base_output = args.output_image
    stem = base_output.stem
    suffix = base_output.suffix or ".png"
    parent = base_output.parent

    for mode in args.render_modes:
        image = _render_scene(
            meshes,
            args.image_size,
            args.camera_elevation,
            args.camera_azimuth,
            args.distance_scale,
            bg_rgba,
            match_view=args.match_image_view,
            mode=mode,
        )
        out_path = parent / f"{stem}_{mode}{suffix}"
        cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
        print(f"Saved {mode} render to {out_path}")


def _build_meshes(
    boxes,
    limbs,
    sections: int,
    min_visibility: float,
) -> List[trimesh.Trimesh]:
    meshes: List[trimesh.Trimesh] = []

    for box in boxes:
        mesh = trimesh.creation.box(extents=box.size)
        axes = np.array(box.axes, dtype=np.float32)
        x_axis = -axes[:, 0]
        y_axis = axes[:, 1]
        z_axis = np.cross(x_axis, y_axis)
        x_axis = _normalize(x_axis)
        y_axis = _normalize(y_axis - np.dot(y_axis, x_axis) * x_axis)
        z_axis = _normalize(z_axis)
        flip_axes = np.column_stack([x_axis, y_axis, z_axis])
        transform = np.eye(4)
        transform[:3, :3] = flip_axes
        transform[:3, 3] = box.center
        mesh.apply_transform(transform)
        color = BOX_COLOR_LOOKUP.get(box.name, (200, 200, 200, 255))
        mesh.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (mesh.vertices.shape[0], 1))
        mesh.metadata = {
            "type": "box",
            "name": box.name,
            "center": box.center.tolist(),
            "axes": axes.tolist(),
            "size": list(box.size),
        }
        meshes.append(mesh)

    for limb in limbs:
        start = limb.start
        end = limb.end
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 1e-4:
            continue

        radius = limb.radius
        cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
        align = trimesh.geometry.align_vectors([0, 0, 1], vector / length)
        cylinder.apply_transform(align)
        midpoint = (start + end) * 0.5
        cylinder.apply_translation(midpoint)

        category = "arm" if "arm" in limb.name else "leg"
        color = CYLINDER_COLORS.get(category, (180, 180, 180, 255))
        cylinder.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (cylinder.vertices.shape[0], 1))
        cylinder.metadata = {
            "type": "limb_segment",
            "name": limb.name,
            "start": start.tolist(),
            "end": end.tolist(),
            "radius": float(radius),
        }
        meshes.append(cylinder)

        joint_labels = LIMB_JOINT_MAP.get(limb.name, ("", ""))
        for endpoint, joint_label in zip((start, end), joint_labels):
            cap = trimesh.creation.icosphere(subdivisions=2, radius=radius * 0.95)
            cap.apply_translation(endpoint)
            cap.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (cap.vertices.shape[0], 1))
            cap.metadata = {"type": "joint", "segment": limb.name, "joint": joint_label}
            meshes.append(cap)

    return meshes


def _render_scene(
    meshes: Sequence[trimesh.Trimesh],
    image_size: int,
    elevation_deg: float,
    azimuth_deg: float,
    distance_scale: float,
    bg_rgba: Tuple[int, int, int, int],
    match_view: bool,
    mode: str,
) -> np.ndarray:
    center, radius = _scene_bounds(meshes)

    if mode in ("orthographic", "wireframe"):
        camera_pose = _front_camera_pose(center, radius, distance_scale)
        return _orthographic_render(
            meshes,
            image_size=image_size,
            camera_pose=camera_pose,
            bg_rgba=bg_rgba,
            wireframe=(mode == "wireframe"),
            toon=False,
            hatch=False,
        )
    if mode == "toon":
        camera_pose = _front_camera_pose(center, radius, distance_scale)
        return _orthographic_render(
            meshes,
            image_size=image_size,
            camera_pose=camera_pose,
            bg_rgba=bg_rgba,
            wireframe=False,
            toon=True,
            hatch=False,
        )
    if mode == "hatch":
        camera_pose = _front_camera_pose(center, radius, distance_scale)
        return _orthographic_render(
            meshes,
            image_size=image_size,
            camera_pose=camera_pose,
            bg_rgba=bg_rgba,
            wireframe=False,
            toon=False,
            hatch=True,
        )

    bg_float = np.array(bg_rgba, dtype=np.float32) / 255.0

    try:
        scene = pyrender.Scene(
            bg_color=bg_float,
            ambient_light=np.array([0.35, 0.35, 0.35]),
        )

        for mesh in meshes:
            pm = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            scene.add(pm)

        if match_view:
            camera_pose = _front_camera_pose(center, radius, distance_scale)
        else:
            camera_pose = _camera_pose(center, radius, elevation_deg, azimuth_deg, distance_scale)
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
        scene.add(camera, pose=camera_pose)

        light_intensity = 3.0
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity)
        scene.add(light, pose=camera_pose)
        side_pose = camera_pose.copy()
        side_pose[:3, 3] = center + np.array([radius * 2.0, radius * 1.1, radius * 1.5])
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=side_pose)

        r = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)
        color, _ = r.render(scene)
        r.delete()
        return color
    except Exception as exc:
        print(f"[render_structure_3d] pyrender offscreen failed ({exc}); falling back to Matplotlib render.")
        return _matplotlib_render(
            meshes,
            image_size=image_size,
            elevation_deg=elevation_deg,
            azimuth_deg=azimuth_deg,
            center=center,
            radius=radius,
            bg_rgba=bg_rgba,
            match_view=match_view,
        )


def _matplotlib_render(
    meshes: Sequence[trimesh.Trimesh],
    image_size: int,
    elevation_deg: float,
    azimuth_deg: float,
    center: np.ndarray,
    radius: float,
    bg_rgba: Tuple[int, int, int, int],
    match_view: bool,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    dpi = 100
    fig = plt.figure(figsize=(image_size / dpi, image_size / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    bg_rgb = np.array(bg_rgba[:3], dtype=np.float32) / 255.0
    fig.patch.set_facecolor(bg_rgb)
    ax.set_facecolor(bg_rgb)

    for mesh in meshes:
        faces = mesh.faces
        vertices = mesh.vertices
        if mesh.visual.kind == "face":
            colors = mesh.visual.face_colors / 255.0
        else:
            colors = mesh.visual.vertex_colors[faces].mean(axis=1) / 255.0
        poly = Poly3DCollection(vertices[faces], facecolors=colors, linewidths=0.4, edgecolor="k", alpha=0.95)
        ax.add_collection3d(poly)

    extent = radius * 1.6
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    if match_view:
        ax.view_init(elev=0.0, azim=180.0)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
    else:
        ax.view_init(elev=elevation_deg, azim=azimuth_deg)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buffer = buffer.reshape(height, width, 4)
    image = buffer[:, :, [1, 2, 3, 0]]
    plt.close(fig)
    return image


def _orthographic_render(
    meshes: Sequence[trimesh.Trimesh],
    image_size: int,
    camera_pose: np.ndarray,
    bg_rgba: Tuple[int, int, int, int],
    wireframe: bool = False,
    toon: bool = False,
    hatch: bool = False,
) -> np.ndarray:
    view = np.eye(4)
    rotation = camera_pose[:3, :3]
    translation = camera_pose[:3, 3]
    view[:3, :3] = rotation.T
    view[:3, 3] = -rotation.T @ translation

    tris: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    intensities: List[np.ndarray] = []

    light_dir = LIGHT_DIRECTION
    cam_forward = camera_pose[:3, 2]
    camera_dir = _normalize(-cam_forward)
    half_vec = _normalize(light_dir + camera_dir)
    ambient = 0.35
    diffuse = 0.6
    rim_gain = 0.35
    spec_gain = 0.25

    for mesh in meshes:
        vertices = mesh.vertices
        ones = np.ones((vertices.shape[0], 1))
        verts_h = np.hstack([vertices, ones])
        verts_cam = (view @ verts_h.T).T[:, :3]

        faces = mesh.faces
        face_vertices = verts_cam[faces]  # (F,3,3)

        if mesh.visual.kind == "face":
            face_colors = mesh.visual.face_colors[:, :3]
        else:
            face_colors = mesh.visual.vertex_colors[faces].mean(axis=1)[:, :3]
        face_colors = face_colors[:, ::-1]  # convert RGB -> BGR for OpenCV

        tris.append(face_vertices)
        colors.append(face_colors)
        normals = mesh.face_normals
        lambert = np.clip(normals @ light_dir, 0.0, 1.0)
        rim = np.clip(1.0 - np.abs(normals @ camera_dir), 0.0, 1.0) ** 2
        spec = np.clip(normals @ half_vec, 0.0, 1.0) ** 18
        intensity = ambient + diffuse * lambert + rim_gain * rim + spec_gain * spec
        intensities.append(np.clip(intensity, 0.1, 1.4))

    if not tris:
        return np.full((image_size, image_size, 4), bg_rgba, dtype=np.uint8)

    tris_arr = np.concatenate(tris, axis=0)
    colors_arr = np.concatenate(colors, axis=0)
    intensity_arr = np.concatenate(intensities, axis=0)

    xs = tris_arr[:, :, 0]
    ys = tris_arr[:, :, 1]

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    span = max(xmax - xmin, ymax - ymin, 1e-5)
    scale = (image_size * 0.8) / span

    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5

    u = (xs - cx) * scale + image_size / 2.0
    v = -(ys - cy) * scale + image_size / 2.0

    depths = tris_arr[:, :, 2].mean(axis=1)
    sort_idx = np.argsort(depths)  # farthest first

    canvas = np.full((image_size, image_size, 3), bg_rgba[:3], dtype=np.uint8)

    hatch_layers: List[np.ndarray] = []

    filled_canvas = np.full((image_size, image_size, 3), bg_rgba[:3], dtype=np.uint8)

    for idx in sort_idx:
        poly = np.stack([u[idx], v[idx]], axis=1).astype(np.int32)
        color = colors_arr[idx].astype(np.float32)

        depth = depths[idx]
        depth_shade = np.clip(
            0.85 + 0.15 * (depth - depths.min()) / (depths.max() - depths.min() + 1e-6),
            0.7,
            1.05,
        )
        shade = intensity_arr[idx] * depth_shade
        shaded_color = np.clip(color * shade, 0, 255).astype(np.uint8)
        if toon:
            shade_levels = np.array([0.35, 0.55, 0.75, 0.95, 1.1], dtype=np.float32)
            nearest = shade_levels[np.abs(shade_levels - shade).argmin()]
            shaded_color = np.clip(color * nearest, 0, 255).astype(np.uint8)

        if wireframe:
            cv2.fillConvexPoly(filled_canvas, poly, tuple(int(c) for c in shaded_color), lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [poly], True, tuple(int(c) for c in shaded_color), thickness=3, lineType=cv2.LINE_AA)
        else:
            cv2.fillConvexPoly(canvas, poly, tuple(int(c) for c in shaded_color), lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [poly], True, (30, 30, 30), thickness=1, lineType=cv2.LINE_AA)

            if hatch:
                hatch_mask = np.zeros((image_size, image_size), dtype=np.uint8)
                cv2.fillConvexPoly(hatch_mask, poly, 255, lineType=cv2.LINE_AA)
                hatch_layers.append(hatch_mask)

    if hatch and hatch_layers:
        combined_mask = np.clip(np.sum(hatch_layers, axis=0), 0, 255).astype(np.uint8)
        stripes = np.zeros_like(canvas, dtype=np.uint8)
        spacing = max(image_size // 90, 4)
        for i in range(0, image_size, spacing):
            cv2.line(stripes, (0, i), (image_size, i - image_size), (40, 40, 40), 1, cv2.LINE_AA)
        stripes = cv2.bitwise_and(stripes, stripes, mask=combined_mask)
        canvas = cv2.addWeighted(canvas, 1.0, stripes, 0.35, 0)
    if wireframe:
        canvas = cv2.addWeighted(filled_canvas, 1.0, canvas, 1.0, 0)

    rgba = np.dstack([canvas, np.full((image_size, image_size), 255, dtype=np.uint8)])
    return rgba


def _scene_bounds(meshes: Sequence[trimesh.Trimesh]) -> Tuple[np.ndarray, float]:
    vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
    center = vertices.mean(axis=0)
    radius = np.linalg.norm(vertices - center, axis=1).max()
    return center, radius


def _camera_pose(
    center: np.ndarray,
    radius: float,
    elevation_deg: float,
    azimuth_deg: float,
    distance_scale: float,
) -> np.ndarray:
    distance = max(radius * distance_scale, 0.5)
    elev = np.deg2rad(elevation_deg)
    azim = np.deg2rad(azimuth_deg)

    eye = center + np.array(
        [
            distance * np.cos(elev) * np.sin(azim),
            -distance * np.cos(elev) * np.cos(azim),
            distance * np.sin(elev),
        ]
    )

    forward = center - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def _front_camera_pose(center: np.ndarray, radius: float, distance_scale: float) -> np.ndarray:
    distance = max(radius * distance_scale, 0.5)
    eye = center - np.array([0.0, 0.0, distance])
    return _pose_from_eye_target(eye, center, np.array([0.0, 1.0, 0.0]))


def _hex_to_rgba(hex_color: str) -> Tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    if len(value) == 6:
        value += "ff"
    if len(value) != 8:
        raise ValueError(f"Expected 6 or 8 hex digits, got '{hex_color}'")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    a = int(value[6:8], 16)
    return r, g, b, a


def _load_pose_from_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(Path(path).read_text())
    count = len(LMS)
    k2d = np.zeros((count, 3), dtype=np.float32)
    k3d = np.zeros((count, 3), dtype=np.float32)
    for member in LMS:
        info = data.get(member.name)
        if info is None:
            raise KeyError(f"Missing landmark {member.name} in {path}")
        k2d[member.value, :2] = info["pixel"]
        k2d[member.value, 2] = info.get("visibility", 1.0)
        k3d[member.value] = info["world"]
    return k2d, k3d


def _pose_from_eye_target(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def _compute_view_alignment(
    k2d: np.ndarray,
    k3d: np.ndarray,
    visibility_thresh: float,
) -> np.ndarray | None:
    mask = k2d[:, 2] > visibility_thresh
    if mask.sum() < 4:
        return None

    pts3d = k3d[mask].copy()
    pts2d = k2d[mask, :2].copy()

    pts2d[:, 1] = -pts2d[:, 1]

    mean3d = pts3d.mean(axis=0)
    mean2d = pts2d.mean(axis=0)

    pts3d_centered = pts3d - mean3d
    pts2d_centered = pts2d - mean2d
    pts2d_padded = np.hstack([pts2d_centered, np.zeros((pts2d_centered.shape[0], 1))])

    H = pts3d_centered.T @ pts2d_padded
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = -R @ mean3d
    return transform


def _clone_meshes_with_modifications(
    meshes: Sequence[trimesh.Trimesh],
    outline_mode: str = "local",
) -> List[trimesh.Trimesh]:
    modified: List[trimesh.Trimesh] = []
    target_joints = {"left_shoulder", "right_shoulder", "left_knee", "right_knee"}
    joint_color = np.array([215, 40, 40, 255], dtype=np.uint8)
    outline_color = np.array([15, 15, 15, 255], dtype=np.uint8)
    spine_points: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    ring_segments = {
        "left_upper_arm",
        "left_forearm",
        "right_upper_arm",
        "right_forearm",
        "left_thigh",
        "left_calf",
        "right_thigh",
        "right_calf",
    }

    for mesh in meshes:
        new_mesh = mesh.copy()
        metadata = getattr(mesh, "metadata", {}) or {}

        if metadata.get("type") == "joint" and metadata.get("joint") in target_joints:
            new_mesh.visual.vertex_colors = np.tile(joint_color, (new_mesh.vertices.shape[0], 1))
        modified.append(new_mesh)

        if metadata.get("type") == "box" and metadata.get("name") in {"pelvis", "ribcage"}:
            outlines = _box_outline_meshes(
                new_mesh,
                outline_color,
                thickness_ratio=0.025,
                diagonals=False,
                mode=outline_mode,
            )
            modified.extend(outlines)
            center_line = _center_line_mesh(new_mesh, joint_color)
            if center_line is not None:
                modified.append(center_line)
            band = _box_mid_band_mesh(new_mesh, joint_color)
            if band is not None:
                modified.append(band)
            center = np.array(new_mesh.metadata.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
            size = np.array(new_mesh.metadata.get("size", [0.1, 0.1, 0.1]), dtype=np.float32)
            spine_points[metadata["name"]] = (center, size)
        if metadata.get("type") == "box" and metadata.get("name") == "head":
            center_line = _center_line_mesh(new_mesh, joint_color)
            if center_line is not None:
                modified.append(center_line)
            band = _box_mid_band_mesh(new_mesh, joint_color)
            if band is not None:
                modified.append(band)
            center = np.array(new_mesh.metadata.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
            size = np.array(new_mesh.metadata.get("size", [0.1, 0.1, 0.1]), dtype=np.float32)
            spine_points[metadata["name"]] = (center, size)

        if metadata.get("type") == "limb_segment" and metadata.get("name") in ring_segments:
            ring = _limb_ring_mesh(metadata, outline_color)
            if ring is not None:
                modified.append(ring)

    spine_order = ["pelvis", "ribcage", "head"]
    spine_centers = [spine_points[name][0] for name in spine_order if name in spine_points]
    if len(spine_centers) >= 2:
        avg_size = np.mean([np.linalg.norm(spine_points[name][1]) for name in spine_order if name in spine_points])
        spine_radius = max(avg_size * 0.04, 0.01)
        spine_mesh = _spine_mesh(spine_centers, spine_radius, joint_color)
        if spine_mesh is not None:
            modified.append(spine_mesh)

    return modified


def _box_outline_meshes(
    box_mesh: trimesh.Trimesh,
    color: np.ndarray,
    thickness_ratio: float = 0.02,
    diagonals: bool = True,
    mode: str = "local",
) -> List[trimesh.Trimesh]:
    outlines: List[trimesh.Trimesh] = []
    meta = getattr(box_mesh, "metadata", {}) or {}
    axes = (
        np.array(meta.get("axes"), dtype=np.float32) if "axes" in meta else None
    )
    size = (
        np.array(meta.get("size"), dtype=np.float32) if "size" in meta else None
    )
    center = (
        np.array(meta.get("center"), dtype=np.float32) if "center" in meta else None
    )

    if mode == "filtered" or axes is None or size is None or center is None:
        edges = box_mesh.edges_unique
        verts = box_mesh.vertices
        extents = box_mesh.extents
        if np.max(extents) <= 1e-6:
            return outlines
        radius = np.max(extents) * thickness_ratio

        for edge in edges:
            start = verts[edge[0]]
            end = verts[edge[1]]
            vec = end - start
            length = np.linalg.norm(vec)
            if length < 1e-6:
                continue
            if not diagonals and not np.any(np.isclose(length, extents, rtol=1e-4, atol=1e-6)):
                continue
            cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
            align = trimesh.geometry.align_vectors([0, 0, 1], vec / length)
            cyl.apply_transform(align)
            cyl.apply_translation((start + end) * 0.5)
            cyl.visual.vertex_colors = np.tile(color, (cyl.vertices.shape[0], 1))
            outlines.append(cyl)
        return outlines

    half = size / 2.0
    local_corners = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], half[1], half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], -half[2]],
            [half[0], half[1], half[2]],
        ],
        dtype=np.float32,
    )
    axis_edge_pairs = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    radius = np.max(size) * thickness_ratio
    axes = np.array(axes)
    for idx_a, idx_b in axis_edge_pairs:
        local_a = local_corners[idx_a]
        local_b = local_corners[idx_b]
        world_a = center + axes @ local_a
        world_b = center + axes @ local_b
        vec = world_b - world_a
        length = np.linalg.norm(vec)
        if length < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
        align = trimesh.geometry.align_vectors([0, 0, 1], vec / length)
        cyl.apply_transform(align)
        cyl.apply_translation((world_a + world_b) * 0.5)
        cyl.visual.vertex_colors = np.tile(color, (cyl.vertices.shape[0], 1))
        outlines.append(cyl)
    return outlines


def _center_line_mesh(box_mesh: trimesh.Trimesh, color: np.ndarray) -> trimesh.Trimesh | None:
    meta = getattr(box_mesh, "metadata", {}) or {}
    axes = (
        np.array(meta.get("axes"), dtype=np.float32) if "axes" in meta else None
    )
    size = (
        np.array(meta.get("size"), dtype=np.float32) if "size" in meta else None
    )
    center = (
        np.array(meta.get("center"), dtype=np.float32) if "center" in meta else None
    )
    if axes is None or size is None or center is None:
        return None
    half_height = size[1] * 0.5
    direction = axes[:, 1]
    start = center - direction * half_height
    end = center + direction * half_height
    vec = end - start
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None
    radius = np.max(size) * 0.02
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
    align = trimesh.geometry.align_vectors([0, 0, 1], vec / length)
    cyl.apply_transform(align)
    cyl.apply_translation((start + end) * 0.5)
    cyl.visual.vertex_colors = np.tile(color, (cyl.vertices.shape[0], 1))
    return cyl


def _box_mid_band_mesh(box_mesh: trimesh.Trimesh, color: np.ndarray) -> trimesh.Trimesh | None:
    meta = getattr(box_mesh, "metadata", {}) or {}
    axes = (
        np.array(meta.get("axes"), dtype=np.float32) if "axes" in meta else None
    )
    size = (
        np.array(meta.get("size"), dtype=np.float32) if "size" in meta else None
    )
    center = (
        np.array(meta.get("center"), dtype=np.float32) if "center" in meta else None
    )
    if axes is None or size is None or center is None:
        return None

    half = size / 2.0
    local_points = []
    for sy, sz in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
        local_points.append(np.array([0.0, sy * half[1], sz * half[2]], dtype=np.float32))
    world_points = [center + axes @ lp for lp in local_points]

    radius = np.max(size) * 0.02
    segments: List[trimesh.Trimesh] = []
    for a, b in zip(world_points, world_points[1:] + world_points[:1]):
        vec = b - a
        length = np.linalg.norm(vec)
        if length < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
        align = trimesh.geometry.align_vectors([0, 0, 1], vec / length)
        cyl.apply_transform(align)
        cyl.apply_translation((a + b) * 0.5)
        cyl.visual.vertex_colors = np.tile(color, (cyl.vertices.shape[0], 1))
        segments.append(cyl)

    if not segments:
        return None
    return trimesh.util.concatenate(segments)


def _limb_ring_mesh(metadata: Dict, color: np.ndarray) -> trimesh.Trimesh | None:
    start = np.array(metadata.get("start", [0.0, 0.0, 0.0]), dtype=np.float32)
    end = np.array(metadata.get("end", [0.0, 0.0, 0.0]), dtype=np.float32)
    radius = float(metadata.get("radius", 0.0))
    vec = end - start
    length = np.linalg.norm(vec)
    if length < 1e-6 or radius <= 0.0:
        return None

    mid = (start + end) * 0.5
    direction = vec / length
    ring_radius = radius * 1.02
    thickness = max(radius * 0.2, 0.01)
    ring = trimesh.creation.cylinder(radius=ring_radius, height=thickness, sections=32)
    align = trimesh.geometry.align_vectors([0, 0, 1], direction)
    ring.apply_transform(align)
    ring.apply_translation(mid)
    ring.visual.vertex_colors = np.tile(color, (ring.vertices.shape[0], 1))
    return ring


def _spine_mesh(points: Sequence[np.ndarray], radius: float, color: np.ndarray) -> trimesh.Trimesh | None:
    if len(points) < 2:
        return None
    segments: List[trimesh.Trimesh] = []
    for a, b in zip(points, points[1:]):
        vec = b - a
        length = np.linalg.norm(vec)
        if length < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
        align = trimesh.geometry.align_vectors([0, 0, 1], vec / length)
        cyl.apply_transform(align)
        cyl.apply_translation((a + b) * 0.5)
        cyl.visual.vertex_colors = np.tile(color, (cyl.vertices.shape[0], 1))
        segments.append(cyl)
    if not segments:
        return None
    return trimesh.util.concatenate(segments)


if __name__ == "__main__":
    main()
