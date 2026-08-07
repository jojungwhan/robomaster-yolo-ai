"""Generate a reference image for the red LEGO cross help-needed pattern."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_help_pattern(cell_pixels=100, border_cells=1):
    grid_size = 5
    image_cells = grid_size + 2 * border_cells
    image = np.full(
        (image_cells * cell_pixels, image_cells * cell_pixels, 3),
        235,
        dtype=np.uint8,
    )
    red = (0, 0, 210)
    cross_cells = {
        (0, 2),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 2),
        (4, 2),
    }
    offset = border_cells * cell_pixels
    gap = max(1, round(cell_pixels * 0.025))
    for row, column in cross_cells:
        x1 = offset + column * cell_pixels + gap
        y1 = offset + row * cell_pixels + gap
        x2 = offset + (column + 1) * cell_pixels - gap
        y2 = offset + (row + 1) * cell_pixels - gap
        cv2.rectangle(image, (x1, y1), (x2, y2), red, -1)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="lego_help_red_cross.png")
    parser.add_argument("--cell-pixels", type=int, default=100)
    args = parser.parse_args()
    if args.cell_pixels < 20:
        raise SystemExit("--cell-pixels must be at least 20.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), generate_help_pattern(args.cell_pixels)):
        raise SystemExit(f"Could not write {output_path}")
    print(f"Saved LEGO help-pattern reference to {output_path.resolve()}")


if __name__ == "__main__":
    main()
