import argparse
from pathlib import Path

import cv2

from lego_vision import generate_marker


def main():
    parser = argparse.ArgumentParser(
        description="Generate an ArUco marker to attach to a LEGO target."
    )
    parser.add_argument("--id", type=int, default=0, dest="marker_id")
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--output", default="lego_marker_0.png")
    args = parser.parse_args()

    if not 0 <= args.marker_id < 50:
        raise SystemExit("Marker ID must be between 0 and 49 for DICT_4X4_50.")
    if args.size < 100:
        raise SystemExit("Marker size must be at least 100 pixels.")

    marker = generate_marker(args.marker_id, args.size)
    margin = max(20, args.size // 8)
    printable = cv2.copyMakeBorder(
        marker,
        margin,
        margin,
        margin,
        margin,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    output_path = Path(args.output).resolve()
    if not cv2.imwrite(str(output_path), printable):
        raise SystemExit(f"Could not write {output_path}")
    print(f"Wrote marker {args.marker_id} to {output_path}")


if __name__ == "__main__":
    main()
