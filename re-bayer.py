from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpeg",
    ".jpg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

VALID_PATTERNS = {"RGGB", "GRBG", "GBRG", "BGGR"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RGB images into RAW Bayer files."
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        default="./",
        help="Folder with input image files, scanned recursively. Default: ./",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        default="./our",
        help=(
            "Folder for output .RAW files. Relative input subfolders are "
            "preserved. Default: ./our"
        ),
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default="RGGB",
        choices=sorted(VALID_PATTERNS),
        type=str.upper,
        help="Bayer pattern. Default: RGGB",
    )
    return parser.parse_args()


def get_input_files(input_folder: Path) -> list[Path]:
    return sorted(
        path
        for path in input_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def pattern_rows(pattern: str) -> tuple[str, str]:
    return pattern[:2], pattern[2:]


def mosaic_channel_for_pixel(x: int, y: int, pattern: str) -> str:
    row0, row1 = pattern_rows(pattern)
    row = row0 if y % 2 == 0 else row1
    return row[0] if x % 2 == 0 else row[1]


def rgb_to_bayer_raw(image: Image.Image, pattern: str) -> bytes:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    pixels = rgb_image.load()
    raw_data = bytearray(width * height)

    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            channel = mosaic_channel_for_pixel(x, y, pattern)
            if channel == "R":
                raw_value = red
            elif channel == "G":
                raw_value = green
            else:
                raw_value = blue

            raw_data[y * width + x] = raw_value

    return bytes(raw_data)


def output_path_for_image(
    image_path: Path,
    input_folder: Path,
    output_folder: Path,
    pattern: str,
    width: int,
    height: int,
) -> Path:
    relative_parent = image_path.relative_to(input_folder).parent
    output_filename = f"{image_path.stem}_{width}x{height}@{pattern}.RAW"
    return output_folder / relative_parent / output_filename


def convert_file(
    image_path: Path,
    input_folder: Path,
    output_folder: Path,
    pattern: str,
) -> Path:
    with Image.open(image_path) as image:
        width, height = image.size
        raw_bytes = rgb_to_bayer_raw(image, pattern)

    output_path = output_path_for_image(
        image_path=image_path,
        input_folder=input_folder,
        output_folder=output_folder,
        pattern=pattern,
        width=width,
        height=height,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw_bytes)
    return output_path


def main() -> int:
    args = parse_args()
    input_folder = Path(args.input_folder).expanduser().resolve()
    output_folder = Path(args.output_folder).expanduser().resolve()

    if not input_folder.exists() or not input_folder.is_dir():
        raise SystemExit(
            f"Input folder does not exist or is not a directory: {input_folder}"
        )

    output_folder.mkdir(parents=True, exist_ok=True)
    input_files = get_input_files(input_folder)

    if not input_files:
        print(f"No supported image files found in {input_folder}")
        return 0

    converted_count = 0
    failed_count = 0

    for image_path in input_files:
        try:
            output_path = convert_file(
                image_path=image_path,
                input_folder=input_folder,
                output_folder=output_folder,
                pattern=args.pattern,
            )
            converted_count += 1
            print(
                f"{image_path.relative_to(input_folder)} -> "
                f"{output_path.relative_to(output_folder)}"
            )
        except Exception as exc:
            failed_count += 1
            print(f"{image_path.relative_to(input_folder)} -> ERROR: {exc}")

    print(
        f"Finished: {converted_count} converted, {failed_count} failed. "
        f"Output folder: {output_folder}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
