"""Split training montage PNGs into top/bottom halves (non-official utility)."""

from __future__ import annotations

import argparse
import os

from PIL import Image


def split_images(image_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    files = [
        f for f in os.listdir(image_dir) if f.endswith(".png") and "result_epoch_" in f
    ]

    for filename in files:
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)
        width, height = img.size

        start_y = int(height * 0.04)
        effective_height = height - start_y
        row_height = effective_height // 10

        top_half = img.crop((0, 0, width, start_y + 5 * row_height))
        bottom_half = img.crop((0, start_y + 5 * row_height, width, height))

        base_name = os.path.splitext(filename)[0]
        top_half.save(os.path.join(output_dir, f"{base_name}_part1.png"))
        bottom_half.save(os.path.join(output_dir, f"{base_name}_part2.png"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", type=str, required=True, help="Directory with result_epoch_*.png")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for splits")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_images(args.image_dir, args.output_dir)
