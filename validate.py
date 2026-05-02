from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
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

STATUS_OK = "OK"
STATUS_ERROR = "ERROR"

ERROR_OK = 0
ERROR_MISSING_RAW = 10
ERROR_RAW_SIZE_MISMATCH = 11
ERROR_IMAGE_SIZE_MISMATCH = 12
ERROR_PSNR_TOO_LOW = 13
ERROR_PROCESSING_FAILED = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate RAW Bayer output by de-Bayering it and computing PSNR "
            "against the source image."
        )
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        default="./",
        help="Folder with source image files, scanned recursively. Default: ./",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        default="./our",
        help=(
            "Folder with output .RAW files. Relative input subfolders are "
            "preserved. Default: ./our"
        ),
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default="RGGB",
        choices=sorted(VALID_PATTERNS),
        type=str.upper,
        help=(
            "Bayer pattern used in the RAW file names and de-Bayer step. "
            "Default: RGGB"
        ),
    )
    parser.add_argument(
        "--min-psnr",
        type=float,
        default=20.0,
        help="Minimum acceptable PSNR in dB. Default: 20.0",
    )
    parser.add_argument(
        "--report-csv",
        default="./validation_report.csv",
        help="CSV report output path. Default: ./validation_report.csv",
    )
    return parser.parse_args()


def get_input_files(input_folder: Path) -> list[Path]:
    return sorted(
        path
        for path in input_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def expected_raw_path(
    image_path: Path,
    input_folder: Path,
    output_folder: Path,
    pattern: str,
) -> Path:
    with Image.open(image_path) as image:
        width, height = image.size
    relative_parent = image_path.relative_to(input_folder).parent
    return (
        output_folder
        / relative_parent
        / f"{image_path.stem}_{width}x{height}@{pattern}.RAW"
    )


OPENCV_BAYER_CONVERSIONS = {
    "RGGB": cv2.COLOR_BayerBG2RGB,
    "GRBG": cv2.COLOR_BayerGB2RGB,
    "GBRG": cv2.COLOR_BayerGR2RGB,
    "BGGR": cv2.COLOR_BayerRG2RGB,
}


def debayer_raw(raw_bytes: bytes, width: int, height: int, pattern: str) -> Image.Image:
    expected_size = width * height
    if len(raw_bytes) != expected_size:
        raise ValueError(
            f"RAW size mismatch: expected {expected_size} bytes for {width}x{height}, got {len(raw_bytes)}."
        )

    bayer_image = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width))
    rgb_array = cv2.cvtColor(bayer_image, OPENCV_BAYER_CONVERSIONS[pattern])
    return Image.fromarray(rgb_array, mode="RGB")


def calculate_psnr(reference: Image.Image, candidate: Image.Image) -> float:
    if reference.size != candidate.size:
        raise ValueError(
            f"Image size mismatch: source {reference.size[0]}x{reference.size[1]}, "
            f"candidate {candidate.size[0]}x{candidate.size[1]}."
        )

    reference_rgb = reference.convert("RGB")
    candidate_rgb = candidate.convert("RGB")

    ref_pixels = reference_rgb.load()
    cand_pixels = candidate_rgb.load()
    width, height = reference_rgb.size

    squared_error_sum = 0.0
    sample_count = width * height * 3

    for y in range(height):
        for x in range(width):
            ref_r, ref_g, ref_b = ref_pixels[x, y]
            cand_r, cand_g, cand_b = cand_pixels[x, y]
            squared_error_sum += (ref_r - cand_r) ** 2
            squared_error_sum += (ref_g - cand_g) ** 2
            squared_error_sum += (ref_b - cand_b) ** 2

    mse = squared_error_sum / sample_count
    if mse == 0:
        return float("inf")

    return 10.0 * math.log10((255.0**2) / mse)


def write_report(report_path: Path, rows: list[dict[str, str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=[
                "input_file",
                "output_file",
                "status",
                "error_code",
                "psnr_db",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_folder = Path(args.input_folder).expanduser().resolve()
    output_folder = Path(args.output_folder).expanduser().resolve()
    report_path = Path(args.report_csv).expanduser().resolve()

    if not input_folder.exists() or not input_folder.is_dir():
        raise SystemExit(
            f"Input folder does not exist or is not a directory: {input_folder}"
        )

    if not output_folder.exists() or not output_folder.is_dir():
        raise SystemExit(
            f"Output folder does not exist or is not a directory: {output_folder}"
        )

    input_files = get_input_files(input_folder)
    if not input_files:
        print(f"No supported image files found in {input_folder}")
        write_report(report_path, [])
        return 0

    report_rows: list[dict[str, str]] = []
    failed_count = 0

    for image_path in input_files:
        input_label = image_path.relative_to(input_folder)
        output_path = expected_raw_path(
            image_path,
            input_folder,
            output_folder,
            args.pattern,
        )
        output_label = output_path.relative_to(output_folder)
        row = {
            "input_file": str(image_path),
            "output_file": str(output_path),
            "status": STATUS_OK,
            "error_code": str(ERROR_OK),
            "psnr_db": "",
            "message": "",
        }

        try:
            if not output_path.exists():
                raise FileNotFoundError(f"RAW output file not found: {output_path}")

            with Image.open(image_path) as source_image:
                source_rgb = source_image.convert("RGB")
                width, height = source_rgb.size

            raw_bytes = output_path.read_bytes()
            debayered = debayer_raw(raw_bytes, width, height, args.pattern)

            if debayered.size != source_rgb.size:
                raise ValueError(
                    f"Image size mismatch: source {source_rgb.size}, "
                    f"debayered {debayered.size}."
                )

            psnr = calculate_psnr(source_rgb, debayered)
            row["psnr_db"] = "inf" if math.isinf(psnr) else f"{psnr:.4f}"

            if psnr < args.min_psnr:
                row["status"] = STATUS_ERROR
                row["error_code"] = str(ERROR_PSNR_TOO_LOW)
                row["message"] = (
                    f"PSNR {psnr:.4f} dB is below minimum threshold {args.min_psnr:.4f} dB."
                )
                failed_count += 1
                print(
                    f"ERROR {ERROR_PSNR_TOO_LOW}: {input_label} vs {output_label} -> "
                    f"PSNR {psnr:.4f} dB below threshold {args.min_psnr:.4f} dB"
                )
            else:
                print(
                    f"OK: {input_label} vs {output_label} -> "
                    f"PSNR {'inf' if math.isinf(psnr) else f'{psnr:.4f}'} dB"
                )

        except FileNotFoundError as exc:
            row["status"] = STATUS_ERROR
            row["error_code"] = str(ERROR_MISSING_RAW)
            row["message"] = str(exc)
            failed_count += 1
            print(f"ERROR {ERROR_MISSING_RAW}: {exc}")
        except ValueError as exc:
            message = str(exc)
            if message.startswith("RAW size mismatch"):
                error_code = ERROR_RAW_SIZE_MISMATCH
            elif message.startswith("Image size mismatch"):
                error_code = ERROR_IMAGE_SIZE_MISMATCH
            else:
                error_code = ERROR_PROCESSING_FAILED
            row["status"] = STATUS_ERROR
            row["error_code"] = str(error_code)
            row["message"] = message
            failed_count += 1
            print(
                f"ERROR {error_code}: {input_label} vs {output_label} -> {message}"
            )
        except Exception as exc:
            row["status"] = STATUS_ERROR
            row["error_code"] = str(ERROR_PROCESSING_FAILED)
            row["message"] = str(exc)
            failed_count += 1
            print(
                f"ERROR {ERROR_PROCESSING_FAILED}: "
                f"{input_label} vs {output_label} -> {exc}"
            )

        report_rows.append(row)

    write_report(report_path, report_rows)
    print(
        f"Validation finished: {len(input_files) - failed_count} passed, {failed_count} failed. "
        f"Report: {report_path}"
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
