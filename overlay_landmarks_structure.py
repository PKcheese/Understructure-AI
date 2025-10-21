"""
Create a combined overlay showing labeled landmarks and coarse structure primitives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapipe as mp

from pose_utils import (
    compute_limb_segments_2d,
    compute_oriented_boxes_2d,
    draw_landmark_labels,
    draw_structure_overlay,
    run_pose_estimation,
)


LMS = mp.solutions.pose.PoseLandmark


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, k2d, _ = run_pose_estimation(args.image)

    boxes_2d = compute_oriented_boxes_2d(k2d)
    limbs_2d = compute_limb_segments_2d(k2d)

    key_labels = [
        (LMS.LEFT_HIP, "left hip"),
        (LMS.RIGHT_HIP, "right hip"),
        (LMS.LEFT_SHOULDER, "left shoulder"),
        (LMS.RIGHT_SHOULDER, "right shoulder"),
        (LMS.LEFT_ELBOW, "left elbow"),
        (LMS.RIGHT_ELBOW, "right elbow"),
        (LMS.LEFT_KNEE, "left knee"),
        (LMS.RIGHT_KNEE, "right knee"),
        (LMS.LEFT_ANKLE, "left ankle"),
        (LMS.RIGHT_ANKLE, "right ankle"),
        (LMS.NOSE, "nose"),
    ]

    # Start by plotting labeled landmarks, then structure on top.
    labeled = draw_landmark_labels(image, k2d, key_labels)
    combined = draw_structure_overlay(
        labeled,
        boxes_2d,
        limbs_2d,
        k2d,
        label_landmarks=False,
    )

    import cv2

    cv2.imwrite(str(args.output), combined)
    print(f"Saved structure overlay to {args.output}")


if __name__ == "__main__":
    main()

