"""
Annotate an image with key pose landmarks (hips, shoulders, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp

from pose_utils import draw_landmark_labels, run_pose_estimation


LMS = mp.solutions.pose.PoseLandmark


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

    landmark_names = [
        (LMS.LEFT_HIP, "left hip"),
        (LMS.RIGHT_HIP, "right hip"),
        (LMS.LEFT_SHOULDER, "left shoulder"),
        (LMS.RIGHT_SHOULDER, "right shoulder"),
        (LMS.LEFT_ELBOW, "left elbow"),
        (LMS.RIGHT_ELBOW, "right elbow"),
        (LMS.LEFT_WRIST, "left wrist"),
        (LMS.RIGHT_WRIST, "right wrist"),
        (LMS.LEFT_KNEE, "left knee"),
        (LMS.RIGHT_KNEE, "right knee"),
        (LMS.LEFT_ANKLE, "left ankle"),
        (LMS.RIGHT_ANKLE, "right ankle"),
        (LMS.NOSE, "nose"),
        (LMS.LEFT_FOOT_INDEX, "left foot"),
        (LMS.RIGHT_FOOT_INDEX, "right foot"),
    ]

    annotated = draw_landmark_labels(
        image, k2d, landmark_names, radius=args.radius, font_scale=args.font_scale
    )
    cv2.imwrite(str(args.output), annotated)
    print(f"Saved labeled landmarks to {args.output}")


if __name__ == "__main__":
    main()

