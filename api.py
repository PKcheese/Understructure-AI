"""
Lightweight REST API that turns an input photo into the stylised maquette GLB.

Usage:
    pip install fastapi uvicorn
    uvicorn api:app --reload

POST /maquette with a multipart field named ``image`` (JPEG or PNG) and the
response will stream back ``maquette_modified_rings.glb``.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from starlette.background import BackgroundTask  # noqa: E402

from pose_utils import (  # noqa: E402
    LMS,
    compute_limb_segments,
    compute_oriented_boxes,
    run_pose_estimation,
)
from render_structure_3d import (  # noqa: E402
    _apply_transform_to_metadata,
    _build_meshes,
    _clone_meshes_with_modifications,
    _compute_view_alignment,
)


app = FastAPI(title="Understructure Maquette API")


def _generate_maquette_glb(image_path: Path, output_path: Path) -> None:
    """Run the existing pipeline and export ``maquette_modified_rings.glb``."""
    image_bgr, k2d, k3d = run_pose_estimation(image_path)

    boxes = compute_oriented_boxes(k3d)
    limbs = compute_limb_segments(k3d)

    alignment = _compute_view_alignment(k2d, k3d, visibility_thresh=0.25)
    meshes = _build_meshes(boxes, limbs, sections=24, min_visibility=0.25)

    if alignment is not None:
        for mesh in meshes:
            mesh.apply_transform(alignment)
            _apply_transform_to_metadata(mesh, alignment)

    modified = _clone_meshes_with_modifications(
        meshes,
        outline_mode="local",  # matches maquette_modified_rings.glb
    )

    combined = trimesh.util.concatenate(modified)
    combined.visual.vertex_colors = np.vstack(
        [mesh.visual.vertex_colors for mesh in modified]
    )
    combined.export(output_path)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"message": "Understructure AI maquette API is running"})


@app.post("/maquette")
async def make_maquette(image: UploadFile = File(...)) -> FileResponse:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are supported.")

    with NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await image.read())

    with NamedTemporaryFile(delete=False, suffix=".glb") as output_tmp:
        output_path = Path(output_tmp.name)

    def cleanup() -> None:
        output_path.unlink(missing_ok=True)
        tmp_path.unlink(missing_ok=True)

    try:
        _generate_maquette_glb(tmp_path, output_path)
    except Exception:
        cleanup()
        raise

    return FileResponse(
        output_path,
        filename="maquette_modified_rings.glb",
        media_type="model/gltf-binary",
        background=BackgroundTask(cleanup),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
