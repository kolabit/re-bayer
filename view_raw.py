from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VALID_BAYER_STARTS = {"RGGB", "GRBG", "GBRG", "BGGR"}
VALID_DEBAYER_METHODS = {"dfpd", "opencv"}
DEFAULT_WINDOW_NAME = "De-Bayered RAW"

#RAW image files should have name like FILENAME_640x480@RGGB.RAW
RAW_SUFFIX_RE = re.compile(
    r"_(?P<width>\d+)x(?P<height>\d+)@(?P<bayer_start>RGGB|GRBG|GBRG|BGGR)$",
    re.IGNORECASE,
)

OPENCV_BAYER_CONVERSIONS = {
    "RGGB": cv2.COLOR_BayerBG2BGR,
    "GRBG": cv2.COLOR_BayerGB2BGR,
    "GBRG": cv2.COLOR_BayerGR2BGR,
    "BGGR": cv2.COLOR_BayerRG2BGR,
}


@dataclass(frozen=True)
class RawImageMetadata:
    width: int
    height: int
    bayer_start: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "View an 8-bit Bayer RAW file after de-Bayering it. "
            "By default, width, height, and Bayer start are read from filenames like "
            "image_500x375@RGGB.RAW."
        )
    )
    parser.add_argument("raw_file", help="Path to the .RAW file to view.")
    parser.add_argument(
        "--width",
        type=int,
        help="Override image width. Default: parsed from filename suffix.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Override image height. Default: parsed from filename suffix.",
    )
    parser.add_argument(
        "-b",
        "--bayer-start",
        choices=sorted(VALID_BAYER_STARTS),
        type=str.upper,
        help="Override Bayer start/pattern. Default: parsed from filename suffix.",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=sorted(VALID_DEBAYER_METHODS),
        default="dfpd",
        help="De-Bayer method: dfpd or opencv. Default: dfpd",
    )
    parser.add_argument(
        "--show-file-name-in-caption",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the RAW file name in the OpenCV window title. Default: enabled",
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=0,
        help="Milliseconds to wait before closing. Use 0 to wait for a key press or window close. Default: 0",
    )
    return parser.parse_args()


def metadata_from_filename(raw_path: Path) -> RawImageMetadata:
    match = RAW_SUFFIX_RE.search(raw_path.stem)
    if not match:
        raise ValueError(
            "Could not parse RAW metadata from filename. Expected suffix like "
            "_500x375@RGGB.RAW, or pass --width, --height, and --bayer-start."
        )

    return RawImageMetadata(
        width=int(match.group("width")),
        height=int(match.group("height")),
        bayer_start=match.group("bayer_start").upper(),
    )


def resolve_metadata(
    raw_path: Path,
    width: int | None,
    height: int | None,
    bayer_start: str | None,
) -> RawImageMetadata:
    try:
        filename_metadata = metadata_from_filename(raw_path)
    except ValueError:
        filename_metadata = None

    resolved_width = width if width is not None else getattr(filename_metadata, "width", None)
    resolved_height = height if height is not None else getattr(filename_metadata, "height", None)
    resolved_bayer_start = (
        bayer_start if bayer_start is not None else getattr(filename_metadata, "bayer_start", None)
    )

    missing = []
    if resolved_width is None:
        missing.append("--width")
    if resolved_height is None:
        missing.append("--height")
    if resolved_bayer_start is None:
        missing.append("--bayer-start")
    if missing:
        raise ValueError(
            f"Missing RAW metadata: {', '.join(missing)}. "
            "Use a filename suffix like _500x375@RGGB.RAW or pass overrides."
        )

    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("Width and height must be positive integers.")

    return RawImageMetadata(
        width=resolved_width,
        height=resolved_height,
        bayer_start=resolved_bayer_start,
    )


def read_bayer_image(raw_path: Path, metadata: RawImageMetadata) -> np.ndarray:
    raw_bytes = raw_path.read_bytes()
    expected_size = metadata.width * metadata.height

    if len(raw_bytes) != expected_size:
        raise ValueError(
            f"RAW size mismatch: expected {expected_size} bytes for "
            f"{metadata.width}x{metadata.height}, got {len(raw_bytes)}."
        )

    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
        (metadata.height, metadata.width)
    )


def debayer_to_bgr(
    bayer_image: np.ndarray,
    bayer_start: str,
    method: str = "dfpd",
) -> np.ndarray:
    if method == "dfpd":
        from debayer_dataset import demosaic_menon2007

        rgb = demosaic_menon2007(bayer_image, bayer_start)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if method != "opencv":
        raise ValueError(f"Unsupported de-Bayer method: {method}")

    return cv2.cvtColor(bayer_image, OPENCV_BAYER_CONVERSIONS[bayer_start])


def window_is_visible(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def wait_until_key_or_close(window_name: str, wait_ms: int) -> None:
    poll_ms = 50
    deadline = None if wait_ms == 0 else time.monotonic() + (wait_ms / 1000.0)

    while window_is_visible(window_name):
        if cv2.waitKey(poll_ms) >= 0:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break


def resolve_window_caption(
    raw_path: Path,
    show_file_name: bool,
) -> str:
    if not show_file_name:
        return DEFAULT_WINDOW_NAME

    return f"{DEFAULT_WINDOW_NAME} - {raw_path.name}"


def resize_window_to_image_scale(
    window_name: str,
    image: np.ndarray,
    scale: float = 1.0,
) -> None:
    target_width = max(1, round(image.shape[1] * scale))
    target_height = max(1, round(image.shape[0] * scale))
    requested_width = target_width
    requested_height = target_height

    for _ in range(5):
        cv2.resizeWindow(window_name, requested_width, requested_height)
        cv2.waitKey(1)

        try:
            _, _, visible_width, visible_height = cv2.getWindowImageRect(window_name)
        except cv2.error:
            return

        if visible_width <= 0 or visible_height <= 0:
            return

        width_delta = target_width - visible_width
        height_delta = target_height - visible_height
        if abs(width_delta) <= 1 and abs(height_delta) <= 1:
            return

        requested_width = max(1, requested_width + width_delta)
        requested_height = max(1, requested_height + height_delta)


def view_raw_file(
    raw_path: Path,
    width: int | None = None,
    height: int | None = None,
    bayer_start: str | None = None,
    method: str = "dfpd",
    show_file_name_in_caption: bool = True,
    wait_ms: int = 0,
) -> None:
    metadata = resolve_metadata(raw_path, width, height, bayer_start)
    bayer_image = read_bayer_image(raw_path, metadata)
    debayered = debayer_to_bgr(bayer_image, metadata.bayer_start, method)
    caption = resolve_window_caption(
        raw_path,
        show_file_name_in_caption,
    )

    cv2.namedWindow(caption, cv2.WINDOW_NORMAL)
    cv2.imshow(caption, debayered)
    resize_window_to_image_scale(caption, debayered)
    wait_until_key_or_close(caption, wait_ms)
    cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    raw_path = Path(args.raw_file).expanduser().resolve()

    if not raw_path.exists() or not raw_path.is_file():
        raise SystemExit(f"RAW file does not exist or is not a file: {raw_path}")

    try:
        view_raw_file(
            raw_path=raw_path,
            width=args.width,
            height=args.height,
            bayer_start=args.bayer_start,
            method=args.method,
            show_file_name_in_caption=args.show_file_name_in_caption,
            wait_ms=args.wait_ms,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
