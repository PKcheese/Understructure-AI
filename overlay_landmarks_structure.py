"""
Create a combined overlay showing labeled landmarks and coarse structure primitives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_utils import (
    compute_label_offsets,
    compute_limb_segments,
    compute_limb_segments_2d,
    compute_oriented_boxes,
    compute_oriented_boxes_2d,
    draw_landmark_labels,
    draw_structure_overlay,
    priority_landmarks,
    run_pose_estimation,
)
from structure_renderer import (
    AlignedStructure,
    composite_overlay,
    render_structure_overlay,
    serialize_box_info,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay pose landmarks and simple structural primitives on an image."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input photo path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("structure_overlay.png"),
        help="Output image path for the combined overlay.",
    )
    parser.add_argument(
        "--output-3d",
        type=Path,
        default=Path("structure_overlay_3d.png"),
        help="Output image path for the pyrender overlay.",
    )
    parser.add_argument(
        "--coords-output",
        type=Path,
        default=Path("structure_boxes_3d.json"),
        help="Path to write 3D box coordinates (pelvis & ribcage).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, k2d, k3d = run_pose_estimation(args.image)

    boxes_2d = compute_oriented_boxes_2d(k2d)
    limbs_2d = compute_limb_segments_2d(k2d)

    key_labels = priority_landmarks()

    overlay_rgba, struct_info = render_structure_overlay(image, k2d, k3d)
    offsets = compute_label_offsets(image.shape)
    combined_3d = composite_overlay(image, overlay_rgba)
    combined_3d = draw_landmark_labels(
        combined_3d,
        k2d,
        key_labels,
        label_offsets=offsets,
        draw_arrows=False,
        draw_text=False,
    )

    combined_2d = draw_landmark_labels(
        image.copy(),
        k2d,
        key_labels,
        label_offsets=offsets,
        draw_arrows=False,
        draw_text=False,
    )
    combined_2d = draw_structure_overlay(
        combined_2d, boxes_2d, limbs_2d, k2d, label_landmarks=False
    )

    import cv2

    cv2.imwrite(str(args.output), combined_2d)
    cv2.imwrite(str(args.output_3d), combined_3d)
    if args.coords_output:
        _write_box_coords(struct_info, args.coords_output)
    print(f"Saved 2D overlay to {args.output}")
    print(f"Saved 3D overlay to {args.output_3d}")
    if args.coords_output:
        print(f"Wrote 3D box coordinates to {args.coords_output}")


def _write_box_coords(struct_info: AlignedStructure, path: Path) -> None:
    data = serialize_box_info(struct_info)
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
