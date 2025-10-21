"""
Generate gesture splines from pose landmarks and overlay them onto the image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pose_utils import build_gesture_curves, draw_gesture_overlay, run_pose_estimation


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
    curves = build_gesture_curves(k2d, iterations=args.gesture_iterations)
    gesture_img = draw_gesture_overlay(image, curves, thickness=args.thickness)

    overlay_path = args.output_dir / "gesture_overlay.png"
    json_path = args.output_dir / "gesture_curves.json"

    cv2.imwrite(str(overlay_path), gesture_img)

    gesture_payload = [
        {"name": curve.name, "points": curve.points.tolist()} for curve in curves
    ]
    json_path.write_text(json.dumps(gesture_payload, indent=2))

    print(f"Saved gesture overlay to {overlay_path}")
    print(f"Saved gesture curves to {json_path}")


if __name__ == "__main__":
    main()
