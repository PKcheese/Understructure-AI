# Photo → Gesture → Structure Toolkit

This folder now exposes the workflow in three progressively richer scripts so you can inspect each stage separately: landmark detection, gesture generation, and structure synthesis. A legacy `gesture_structure_pipeline.py` still runs the full chain in one go if you prefer.

## Example Output

![Gesture to maquette overlay](docs/example_image.png)

The example above comes straight from the REST API using the default gesture and structure passes. It shows the annotated source photo alongside the generated 3D maquette overlay, so you can quickly sanity-check your setup before running the full pipeline.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## REST API

```bash
uvicorn api:app --reload
```

Then send a multipart form-data POST to `http://127.0.0.1:8000/maquette` with an `image` field containing a JPEG or PNG. The response streams back `maquette_modified_rings.glb`. You can exercise the endpoint via the interactive docs (`/docs`) or the helper client in `import requests.py`.

## 1. Mark landmarks

```bash
python annotate_landmarks.py \
  --image input.jpg \
  --output outputs/landmarks.png
```

Draws circles and labels (hips, shoulders, elbows, etc.) on the source image so you can verify detection quality.

## 2. Build gesture curves

```bash
python generate_gesture.py \
  --image input.jpg \
  --output-dir outputs/gesture \
  --gesture-iterations 3 \
  --thickness 3
```

Produces `gesture_overlay.png` plus `gesture_curves.json` describing the smoothed splines.

## 3. Create structure primitives

```bash
python build_structure.py \
  --image input.jpg \
  --output-dir outputs/structure \
  --save-landmarks
```

Exports `structure.json`, `structure.obj`, and (optionally) the raw `landmarks.npz`.

## 4. Combined overlay (landmarks + structure)

```bash
python overlay_landmarks_structure.py \
  --image input.jpg \
  --output outputs/structure_overlay_2d.png \
  --output-3d outputs/structure_overlay_3d.png \
  --coords-output outputs/structure_boxes_3d.json
```

Creates both annotated renders: the original 2D capsule/box overlay (`structure_overlay_2d.png`) plus the pyrender/trimesh shaded rig (`structure_overlay_3d.png`). The accompanying `structure_boxes_3d.json` stores the aligned pelvis and ribcage box coordinates in image space (including transform parameters).

## Optional: one-shot pipeline

```bash
python gesture_structure_pipeline.py \
  --image input.jpg \
  --output-dir outputs/full_run
```

This now also emits both `structure_overlay_2d.png` (capsule/box overlay) and `structure_overlay_3d.png` (trimesh+pyrender) alongside the earlier gesture/structure exports, plus `structure_boxes_3d.json` containing the aligned pelvis/ribcage coordinates. The OBJ output uses low-poly cylinders for limbs instead of simple lines.

### Compare multiple 3D styles

```bash
python generate_3d_variants.py \
  --image input.jpg \
  --output-dir outputs/variants
```

Writes `structure_overlay_3d_output1.png` (painterly), `structure_overlay_3d_output2.png` (solid), and `structure_overlay_3d_output3.png` (sketch) so you can pick the foreshortening/occlusion treatment that works best.

## Under the hood

1. [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose) extracts 33 landmarks in image and “world” coordinates.
2. Gesture splines use Chaikin smoothing over line-of-action and limb chains.
3. Structure boxes derive from hip/shoulder frames and simple scale heuristics; limbs become cylinders encoded as segments.
4. Shaded overlays are rendered with trimesh primitives through pyrender, using a camera estimated from landmark correspondences.

Tweak chain definitions or scaling heuristics in `pose_utils.py`, or customize lighting/material settings in `structure_renderer.py`, to suit your style.
