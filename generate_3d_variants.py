"""Produce multiple 3D overlay styles for comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from pose_utils import (
    compute_label_offsets,
    draw_landmark_labels,
    priority_landmarks,
    run_pose_estimation,
)
from structure_renderer import composite_overlay, render_structure_overlay


STYLE_PRESETS = [
    ("painterly", "structure_overlay_3d_output1.png"),
    ("solid", "structure_overlay_3d_output2.png"),
    ("sketch", "structure_overlay_3d_output3.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple 3D overlay variants.")
    parser.add_argument("--image", type=Path, required=True, help="Input photo path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save overlays.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image, k2d, k3d = run_pose_estimation(args.image)
    offsets = compute_label_offsets(image.shape)
    labels = priority_landmarks()

    for idx, (style, filename) in enumerate(STYLE_PRESETS, start=1):
        overlay_rgba, _ = render_structure_overlay(image, k2d, k3d, style=style)
        composite = composite_overlay(image, overlay_rgba)
        composite = draw_landmark_labels(
            composite,
            k2d,
            labels,
            label_offsets=offsets,
            draw_arrows=False,
            draw_text=False,
        )
        out_path = args.output_dir / filename
        cv2.imwrite(str(out_path), composite)
        print(f"[{idx}] Saved {style} overlay to {out_path}")


if __name__ == "__main__":
    main()
