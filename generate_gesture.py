"""
Generate gesture splines from pose landmarks and overlay them onto the image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from pose_utils import build_gesture_curves, draw_gesture_overlay, run_pose_estimation


def generate_gesture_assets(
    image: np.ndarray,
    landmarks_2d: np.ndarray,
    output_dir: Path,
    *,
    iterations: int = 3,
    thickness: int = 3,
    export_curves: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """
    Render the gesture overlay (and optionally curves JSON) into ``output_dir``.

    Returns a tuple of (overlay_path, curves_json_path_or_None).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = build_gesture_curves(landmarks_2d, iterations=iterations)
    gesture_img = draw_gesture_overlay(image, curves, thickness=thickness)

    overlay_path = output_dir / "gesture_overlay.png"
    cv2.imwrite(str(overlay_path), gesture_img)

    curves_path: Optional[Path] = None
    if export_curves:
        curves_path = output_dir / "gesture_curves.json"
        payload = [
            {"name": curve.name, "points": curve.points.tolist()} for curve in curves
        ]
        curves_path.write_text(json.dumps(payload, indent=2))

    return overlay_path, curves_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create gesture lines from a photo.")
    parser.add_argument("--image", type=Path, required=True, help="Input photo path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store gesture outputs.",
    )
    parser.add_argument(
        "--gesture-iterations",
        type=int,
        default=3,
        help="Chaikin smoothing iterations for gesture curves.",
    )
    parser.add_argument(
        "--thickness", type=int, default=3, help="Stroke thickness for the overlay."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image, k2d, _ = run_pose_estimation(args.image)
    overlay_path, json_path = generate_gesture_assets(
        image,
        k2d,
        args.output_dir,
        iterations=args.gesture_iterations,
        thickness=args.thickness,
        export_curves=True,
    )

    print(f"Saved gesture overlay to {overlay_path}")
    if json_path is not None:
        print(f"Saved gesture curves to {json_path}")


if __name__ == "__main__":
    main()
