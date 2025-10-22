"""
Legacy convenience script that still runs the full pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pose_utils import (
    build_gesture_curves,
    compute_label_offsets,
    compute_limb_segments,
    compute_limb_segments_2d,
    compute_oriented_boxes,
    compute_oriented_boxes_2d,
    draw_gesture_overlay,
    draw_landmark_labels,
    draw_structure_overlay,
    export_structure_json,
    export_structure_obj,
    priority_landmarks,
    run_pose_estimation,
    smooth_curve,
)
from structure_renderer import composite_overlay, render_structure_overlay, serialize_box_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Photo → Gesture → Structure pipeline.")
    parser.add_argument("--image", type=Path, required=True, help="Input photo path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--gesture-iterations",
        type=int,
        default=3,
        help="Chaikin smoothing iterations for gesture curves.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=3,
        help="Stroke thickness (px) for gesture overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image, k2d, k3d = run_pose_estimation(args.image)

    curves = build_gesture_curves(k2d, iterations=args.gesture_iterations)
    gesture_img = draw_gesture_overlay(image, curves, thickness=args.thickness)

    boxes = compute_oriented_boxes(k3d)
    limbs = compute_limb_segments(k3d)
    key_labels = priority_landmarks()

    cv2.imwrite(str(args.output_dir / "gesture_overlay.png"), gesture_img)
    export_structure_json(boxes, limbs, args.output_dir / "structure.json")
    export_structure_obj(boxes, limbs, args.output_dir / "structure.obj")
    np.savez(
        args.output_dir / "landmarks.npz",
        landmarks_2d=k2d,
        landmarks_3d=k3d,
    )

    overlay_rgba, struct_info = render_structure_overlay(image, k2d, k3d)
    offsets = compute_label_offsets(image.shape)
    structure_overlay_3d = composite_overlay(image, overlay_rgba)
    structure_overlay_3d = draw_landmark_labels(
        structure_overlay_3d,
        k2d,
        key_labels,
        label_offsets=offsets,
        draw_arrows=False,
        draw_text=False,
    )
    cv2.imwrite(
        str(args.output_dir / "structure_overlay_3d.png"), structure_overlay_3d
    )

    coords_path = args.output_dir / "structure_boxes_3d.json"
    coords_path.write_text(json.dumps(serialize_box_info(struct_info), indent=2))

    structure_overlay_2d = draw_landmark_labels(
        image.copy(),
        k2d,
        key_labels,
        label_offsets=offsets,
        draw_arrows=False,
        draw_text=False,
    )
    boxes_2d = compute_oriented_boxes_2d(k2d)
    limbs_2d = compute_limb_segments_2d(k2d)
    structure_overlay_2d = draw_structure_overlay(
        structure_overlay_2d, boxes_2d, limbs_2d, k2d, label_landmarks=False
    )
    cv2.imwrite(
        str(args.output_dir / "structure_overlay_2d.png"), structure_overlay_2d
    )

    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
