"""
Lightweight REST API that turns an input photo into the stylised maquette GLB.

Usage:
    pip install fastapi uvicorn
    uvicorn api:app --reload

POST /maquette with a multipart field named ``image`` (JPEG or PNG) and the
response will stream back ``maquette_variants.zip`` containing the
aligned, non-aligned, and stabilized maquette GLBs, a landmark-only export, and a preview PNG.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import List, Optional, Sequence

import cv2  # noqa: E402

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from fastapi import FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from starlette.background import BackgroundTask  # noqa: E402

from pose_utils import (  # noqa: E402
    LMS,
    compute_limb_segments,
    compute_oriented_boxes,
    run_pose_estimation,
)
from stabilize_landmarks import PoseStabilizer  # noqa: E402
from render_structure_3d import (  # noqa: E402
    _apply_transform_to_metadata,
    _build_meshes,
    _clone_meshes_with_modifications,
    _compute_view_alignment,
    _render_scene,
)


app = FastAPI(title="Understructure Maquette API")


BASE_EXPORT_TRANSFORM = np.eye(4, dtype=np.float32)
BASE_EXPORT_TRANSFORM[1, 1] = -1.0
BASE_EXPORT_TRANSFORM[2, 2] = -1.0


def _apply_mesh_transform(meshes: Sequence[trimesh.Trimesh], transform: np.ndarray) -> None:
    for mesh in meshes:
        mesh.apply_transform(transform)
        _apply_transform_to_metadata(mesh, transform)


def _build_landmark_markers(k3d: np.ndarray) -> Sequence[trimesh.Trimesh]:
    """Return small landmark markers (spheres) for each MediaPipe landmark."""
    coords = np.array([k3d[member.value] for member in LMS], dtype=np.float32)
    if coords.size == 0:
        return []

    center = coords.mean(axis=0)
    max_radius = np.linalg.norm(coords - center, axis=1).max(initial=0.0)
    marker_radius = max(float(max_radius) * 0.03, 0.01)
    color = np.array([245, 180, 65, 255], dtype=np.uint8)

    markers: List[trimesh.Trimesh] = []
    for member, coord in zip(LMS, coords):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=marker_radius)
        sphere.apply_translation(coord)
        sphere.visual.vertex_colors = np.tile(color, (sphere.vertices.shape[0], 1))
        sphere.metadata = {
            "type": "landmark",
            "name": member.name.lower(),
            "index": member.value,
            "position": coord.tolist(),
        }
        markers.append(sphere)
    return markers


def _generate_maquette_assets(image_path: Path, output_dir: Path) -> dict[str, Path]:
    """Run the pipeline and export assets into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr, k2d, k3d = run_pose_estimation(image_path)
    stabilizer = PoseStabilizer(ema_alpha=0.35)
    try:
        k3d_stabilized = stabilizer.stabilize(k3d, return_world=True)
        if not np.all(np.isfinite(k3d_stabilized)):
            raise ValueError("Stabilized landmarks contain non-finite values.")
    except Exception:
        print("Stabilizer failed; falling back to raw landmarks for this frame.", flush=True)
        k3d_stabilized = None

    boxes = compute_oriented_boxes(k3d)
    limbs, hand_boxes = compute_limb_segments(k3d)
    boxes = boxes + hand_boxes
    landmark_meshes = _build_landmark_markers(k3d)

    alignment = _compute_view_alignment(k2d, k3d, visibility_thresh=0.25)
    meshes = _build_meshes(boxes, limbs, sections=24, min_visibility=0.25)

    meshes_stabilized = None
    alignment_stabilized = None
    if k3d_stabilized is not None:
        boxes_stabilized = compute_oriented_boxes(k3d_stabilized)
        limbs_stabilized, hand_boxes_stabilized = compute_limb_segments(k3d_stabilized)
        boxes_stabilized = boxes_stabilized + hand_boxes_stabilized
        alignment_stabilized = _compute_view_alignment(k2d, k3d_stabilized, visibility_thresh=0.25)
        meshes_stabilized = _build_meshes(
            boxes_stabilized, limbs_stabilized, sections=24, min_visibility=0.25
        )

    def _export_variant(meshes_variant: Sequence[trimesh.Trimesh], target: Path) -> None:
        modified = _clone_meshes_with_modifications(
            meshes_variant,
            outline_mode="local",
        )
        combined = trimesh.util.concatenate(modified)
        combined.visual.vertex_colors = np.vstack(
            [mesh.visual.vertex_colors for mesh in modified]
        )
        combined.export(target)

    no_match_meshes = [mesh.copy() for mesh in meshes]
    _apply_mesh_transform(no_match_meshes, BASE_EXPORT_TRANSFORM)
    no_match_path = output_dir / "maquette_modified_rings_nomatch.glb"
    _export_variant(no_match_meshes, no_match_path)

    match_meshes = [mesh.copy() for mesh in meshes]
    if alignment is not None:
        for mesh in match_meshes:
            mesh.apply_transform(alignment)
            _apply_transform_to_metadata(mesh, alignment)

    stabilized_path: Optional[Path] = None
    if meshes_stabilized is not None:
        stabilized_meshes = [mesh.copy() for mesh in meshes_stabilized]
        if alignment_stabilized is not None:
            for mesh in stabilized_meshes:
                mesh.apply_transform(alignment_stabilized)
                _apply_transform_to_metadata(mesh, alignment_stabilized)
        _apply_mesh_transform(stabilized_meshes, BASE_EXPORT_TRANSFORM)
        stabilized_path = output_dir / "maquette_with_stabilization.glb"
        _export_variant(stabilized_meshes, stabilized_path)

    preview_meshes = [mesh.copy() for mesh in match_meshes]
    _apply_mesh_transform(preview_meshes, BASE_EXPORT_TRANSFORM)
    preview_rgba = _render_scene(
        preview_meshes,
        image_size=960,
        elevation_deg=35.0,
        azimuth_deg=25.0,
        distance_scale=3.0,
        bg_rgba=(242, 242, 243, 255),
        match_view=False,
        mode="matplotlib",
    )
    preview_bgra = cv2.cvtColor(preview_rgba, cv2.COLOR_RGBA2BGRA)
    preview_path = output_dir / "maquette_preview.png"
    cv2.imwrite(str(preview_path), preview_bgra)

    _apply_mesh_transform(match_meshes, BASE_EXPORT_TRANSFORM)
    match_path = output_dir / "maquette_modified_rings.glb"
    _export_variant(match_meshes, match_path)

    landmark_meshes_copy = [mesh.copy() for mesh in landmark_meshes]
    _apply_mesh_transform(landmark_meshes_copy, BASE_EXPORT_TRANSFORM)
    landmarks_path = output_dir / "mediapipe_landmarks.glb"
    _export_variant(landmark_meshes_copy, landmarks_path)

    zip_path = output_dir / "maquette_variants.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(no_match_path, arcname=no_match_path.name)
        zf.write(match_path, arcname=match_path.name)
        if stabilized_path is not None:
            zf.write(stabilized_path, arcname=stabilized_path.name)
        zf.write(landmarks_path, arcname=landmarks_path.name)
        zf.write(preview_path, arcname=preview_path.name)

    return {
        "zip": zip_path,
        "no_match": no_match_path,
        "match": match_path,
        "stabilized": stabilized_path,
        "landmarks": landmarks_path,
        "preview": preview_path,
    }


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"message": "Understructure AI maquette API is running"})


@app.post("/maquette")
async def make_maquette(request: Request, image: UploadFile = File(...)) -> FileResponse:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are supported.")

    with NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await image.read())

    output_tmpdir = TemporaryDirectory()
    output_dir = Path(output_tmpdir.name)

    def cleanup() -> None:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        output_tmpdir.cleanup()

    try:
        assets = _generate_maquette_assets(tmp_path, output_dir)
    except Exception:
        cleanup()
        raise

    variant = (request.query_params.get("variant") or "zip").lower()
    if variant in {"nomatch", "nomatch_glb"}:
        response_path = assets["no_match"]
        filename = "maquette_modified_rings_nomatch.glb"
        media_type = "model/gltf-binary"
    elif variant == "match":
        response_path = assets["match"]
        filename = "maquette_modified_rings.glb"
        media_type = "model/gltf-binary"
    elif variant == "preview":
        response_path = assets["preview"]
        filename = "maquette_preview.png"
        media_type = "image/png"
    elif variant == "landmarks":
        response_path = assets["landmarks"]
        filename = "mediapipe_landmarks.glb"
        media_type = "model/gltf-binary"
    elif variant in {"stabilized", "stabilized_glb"}:
        response_path = assets["stabilized"]
        if response_path is None:
            cleanup()
            raise HTTPException(status_code=503, detail="Stabilized variant unavailable for this frame.")
        filename = "maquette_with_stabilization.glb"
        media_type = "model/gltf-binary"
    else:
        response_path = assets["zip"]
        filename = "maquette_variants.zip"
        media_type = "application/zip"

    return FileResponse(
        response_path,
        filename=filename,
        media_type=media_type,
        background=BackgroundTask(cleanup),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
