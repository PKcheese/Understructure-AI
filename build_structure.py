"""
Construct coarse structural primitives (boxes + limb segments) from pose landmarks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pose_utils import (
    compute_limb_segments,
    compute_oriented_boxes,
    export_structure_json,
    export_structure_obj,
    run_pose_estimation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate oriented boxes and limb segments from a photo."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input photo path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for structure exports.",
    )
    parser.add_argument(
        "--save-landmarks",
        action="store_true",
        help="Also save raw pose landmarks to landmarks.npz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, k2d, k3d = run_pose_estimation(args.image)
    boxes = compute_oriented_boxes(k3d)
    limbs = compute_limb_segments(k3d)

    json_path = args.output_dir / "structure.json"
    obj_path = args.output_dir / "structure.obj"
    export_structure_json(boxes, limbs, json_path)
    export_structure_obj(boxes, limbs, obj_path)

    if args.save_landmarks:
        np.savez(args.output_dir / "landmarks.npz", landmarks_2d=k2d, landmarks_3d=k3d)

    print(f"Saved structure JSON to {json_path}")
    print(f"Saved structure OBJ to {obj_path}")
    if args.save_landmarks:
        print(f"Saved landmarks to {args.output_dir / 'landmarks.npz'}")


if __name__ == "__main__":
    main()

