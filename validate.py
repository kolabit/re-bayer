from __future__ import annotations

import argparse
import csv
import math
import shlex
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
        default="./out",
        help=(
            "Folder with output .RAW files. Relative input subfolders are "
            "preserved. Default: ./out"
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
        "-m",
        "--min-psnr",
        type=float,
        default=15.0,
        help="Minimum acceptable PSNR in dB. Default: 15.0",
    )
    parser.add_argument(
        "--dfpd-recheck-margin-percent",
        type=float,
        default=15.0,
        help=(
            "Re-check fast PSNR failures with DFPD when the fast PSNR is within "
            "this percent below --min-psnr. Use 0 to re-check only exact-threshold "
            "failures. Default: 15.0"
        ),
    )
    parser.add_argument(
        "-r",
        "--report-csv",
        default="./validation_report.csv",
        help="CSV report output path. Default: ./validation_report.csv",
    )
    parser.add_argument(
        "-e",
        "--psnr-error-view-script",
        default="./check_err_images.sh",
        help=(
            "Optional shell script output path. When set, writes commands that run "
            "view_raw.py for each RAW file that fails the PSNR threshold. "
            "Default: ./check_err_images.sh"
        ),
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
    bayer_image = raw_bytes_to_bayer(raw_bytes, width, height)
    rgb_array = cv2.cvtColor(bayer_image, OPENCV_BAYER_CONVERSIONS[pattern])
    return Image.fromarray(rgb_array, mode="RGB")


def raw_bytes_to_bayer(raw_bytes: bytes, width: int, height: int) -> np.ndarray:
    expected_size = width * height
    if len(raw_bytes) != expected_size:
        raise ValueError(
            f"RAW size mismatch: expected {expected_size} bytes for {width}x{height}, got {len(raw_bytes)}."
        )

    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width))


def debayer_raw_dfpd(
    raw_bytes: bytes,
    width: int,
    height: int,
    pattern: str,
) -> Image.Image:
    from debayer_dataset import demosaic_menon2007

    bayer_image = raw_bytes_to_bayer(raw_bytes, width, height)
    rgb_array = demosaic_menon2007(bayer_image, pattern)
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


def format_psnr(psnr: float) -> str:
    return "inf" if math.isinf(psnr) else f"{psnr:.4f}"


def should_recheck_with_dfpd(
    fast_psnr: float,
    min_psnr: float,
    margin_percent: float,
) -> bool:
    if math.isinf(fast_psnr) or fast_psnr >= min_psnr:
        return False

    recheck_floor = min_psnr * (1.0 - (margin_percent / 100.0))
    return fast_psnr >= recheck_floor


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
                "fast_psnr_db",
                "dfpd_recheck_psnr_db",
                "dfpd_recheck",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_view_script(script_path: Path, raw_paths: list[Path]) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    view_raw_path = Path(__file__).resolve().with_name("view_raw.py")
    lines = [
        "#!/bin/sh",
        "set -e",
        "",
        "# Generated by validate.py for RAW files that failed PSNR validation.",
    ]

    if raw_paths:
        lines.extend(
            f"python3 {shlex.quote(str(view_raw_path))} {shlex.quote(str(raw_path))}"
            for raw_path in raw_paths
        )
    else:
        lines.append("echo 'No errors found'")

    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | 0o111)


def main() -> int:
    args = parse_args()
    input_folder = Path(args.input_folder).expanduser().resolve()
    output_folder = Path(args.output_folder).expanduser().resolve()
    report_path = Path(args.report_csv).expanduser().resolve()
    psnr_error_view_script = (
        Path(args.psnr_error_view_script).expanduser().resolve()
        if args.psnr_error_view_script
        else None
    )

    if not input_folder.exists() or not input_folder.is_dir():
        raise SystemExit(
            f"Input folder does not exist or is not a directory: {input_folder}"
        )

    if not output_folder.exists() or not output_folder.is_dir():
        raise SystemExit(
            f"Output folder does not exist or is not a directory: {output_folder}"
        )

    if args.dfpd_recheck_margin_percent < 0:
        raise SystemExit("--dfpd-recheck-margin-percent must be non-negative.")

    input_files = get_input_files(input_folder)
    if not input_files:
        print(f"No supported image files found in {input_folder}")
        write_report(report_path, [])
        if psnr_error_view_script is not None:
            write_view_script(psnr_error_view_script, [])
            print(f"PSNR error view script: {psnr_error_view_script}")
        return 0

    report_rows: list[dict[str, str]] = []
    psnr_error_raw_paths: list[Path] = []
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
            "fast_psnr_db": "",
            "dfpd_recheck_psnr_db": "",
            "dfpd_recheck": "not_needed",
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
            row["psnr_db"] = format_psnr(psnr)
            row["fast_psnr_db"] = format_psnr(psnr)

            if psnr < args.min_psnr:
                if should_recheck_with_dfpd(
                    psnr,
                    args.min_psnr,
                    args.dfpd_recheck_margin_percent,
                ):
                    row["dfpd_recheck"] = "attempted"
                    try:
                        dfpd_debayered = debayer_raw_dfpd(
                            raw_bytes,
                            width,
                            height,
                            args.pattern,
                        )
                        dfpd_psnr = calculate_psnr(source_rgb, dfpd_debayered)
                        row["dfpd_recheck_psnr_db"] = format_psnr(dfpd_psnr)

                        if dfpd_psnr >= args.min_psnr:
                            row["dfpd_recheck"] = "passed"
                            row["psnr_db"] = format_psnr(dfpd_psnr)
                            row["message"] = (
                                f"Fast PSNR {psnr:.4f} dB was below minimum threshold "
                                f"{args.min_psnr:.4f} dB; DFPD re-check passed with "
                                f"{format_psnr(dfpd_psnr)} dB."
                            )
                            print(
                                f"OK: {input_label} vs {output_label} -> "
                                f"fast PSNR {psnr:.4f} dB, DFPD re-check "
                                f"{format_psnr(dfpd_psnr)} dB"
                            )
                        else:
                            row["dfpd_recheck"] = "failed"
                            row["status"] = STATUS_ERROR
                            row["error_code"] = str(ERROR_PSNR_TOO_LOW)
                            row["message"] = (
                                f"Fast PSNR {psnr:.4f} dB and DFPD re-check PSNR "
                                f"{dfpd_psnr:.4f} dB are below minimum threshold "
                                f"{args.min_psnr:.4f} dB."
                            )
                            psnr_error_raw_paths.append(output_path)
                            failed_count += 1
                            print(
                                f"ERROR {ERROR_PSNR_TOO_LOW}: {input_label} vs {output_label} -> "
                                f"fast PSNR {psnr:.4f} dB, DFPD re-check "
                                f"{dfpd_psnr:.4f} dB below threshold {args.min_psnr:.4f} dB"
                            )
                    except Exception as exc:
                        row["dfpd_recheck"] = "error"
                        row["status"] = STATUS_ERROR
                        row["error_code"] = str(ERROR_PSNR_TOO_LOW)
                        row["message"] = (
                            f"Fast PSNR {psnr:.4f} dB is below minimum threshold "
                            f"{args.min_psnr:.4f} dB; DFPD re-check failed: {exc}"
                        )
                        psnr_error_raw_paths.append(output_path)
                        failed_count += 1
                        print(
                            f"ERROR {ERROR_PSNR_TOO_LOW}: {input_label} vs {output_label} -> "
                            f"fast PSNR {psnr:.4f} dB below threshold {args.min_psnr:.4f} dB; "
                            f"DFPD re-check failed: {exc}"
                        )
                else:
                    row["dfpd_recheck"] = "skipped"
                    row["status"] = STATUS_ERROR
                    row["error_code"] = str(ERROR_PSNR_TOO_LOW)
                    row["message"] = (
                        f"PSNR {psnr:.4f} dB is below minimum threshold {args.min_psnr:.4f} dB "
                        f"and outside the {args.dfpd_recheck_margin_percent:.4f}% DFPD "
                        "re-check margin."
                    )
                    psnr_error_raw_paths.append(output_path)
                    failed_count += 1
                    print(
                        f"ERROR {ERROR_PSNR_TOO_LOW}: {input_label} vs {output_label} -> "
                        f"PSNR {psnr:.4f} dB below threshold {args.min_psnr:.4f} dB "
                        "and outside DFPD re-check margin"
                    )
            else:
                print(
                    f"OK: {input_label} vs {output_label} -> "
                    f"PSNR {format_psnr(psnr)} dB"
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
    if psnr_error_view_script is not None:
        write_view_script(psnr_error_view_script, psnr_error_raw_paths)
        print(
            f"PSNR error view script: {psnr_error_view_script} "
            f"({len(psnr_error_raw_paths)} RAW files)"
        )
    print(
        f"Validation finished: {len(input_files) - failed_count} passed, {failed_count} failed. "
        f"Report: {report_path}"
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
