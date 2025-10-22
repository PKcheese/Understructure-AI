"""
Annotate an image with key pose landmarks (hips, shoulders, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from pose_utils import compute_label_offsets, draw_landmark_labels, priority_landmarks, run_pose_estimation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mark pose landmarks on an image.")
    parser.add_argument("--image", type=Path, required=True, help="Input photo path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("landmarks_overlay.png"),
        help="Output image path for the labeled landmarks.",
    )
    parser.add_argument(
        "--font-scale", type=float, default=0.6, help="Font scale for labels."
    )
    parser.add_argument(
        "--radius", type=int, default=6, help="Circle radius for landmark markers."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, k2d, _ = run_pose_estimation(args.image)

    landmark_names = priority_landmarks()
    offsets = compute_label_offsets(image.shape)

    annotated = draw_landmark_labels(
        image,
        k2d,
        landmark_names,
        radius=args.radius,
        font_scale=args.font_scale,
        label_offsets=offsets,
        draw_arrows=False,
        draw_text=False,
    )
    cv2.imwrite(str(args.output), annotated)
    print(f"Saved labeled landmarks to {args.output}")


if __name__ == "__main__":
    main()
