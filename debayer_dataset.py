#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

#RAW image files should have name like FILENAME_640x480@RGGB.RAW
RAW_RE = re.compile(
    r"^(?P<name>.+)_(?P<w>\d+)x(?P<h>\d+)@(?P<pattern>RGGB|BGGR|GRBG|GBRG)\.RAW$",
    re.IGNORECASE,
)


def resize_bayer_plane(plane: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_height, target_width = size
    image = Image.fromarray(plane, mode="L")
    resized = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=plane.dtype)


def resize_bayer_planes(bayer: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(size, int):
        target_height, target_width = size, size
    else:
        target_height, target_width = size

    if target_height % 2 != 0 or target_width % 2 != 0:
        raise ValueError("Bayer resize height and width must be even.")
    if bayer.shape[0] % 2 != 0 or bayer.shape[1] % 2 != 0:
        raise ValueError("Bayer RAW height and width must be even.")

    p00 = bayer[0::2, 0::2]
    p01 = bayer[0::2, 1::2]
    p11 = bayer[1::2, 1::2]
    p10 = bayer[1::2, 0::2]

    plane_size = (target_height // 2, target_width // 2)
    p00 = resize_bayer_plane(p00, plane_size)
    p01 = resize_bayer_plane(p01, plane_size)
    p11 = resize_bayer_plane(p11, plane_size)
    p10 = resize_bayer_plane(p10, plane_size)

    resized = np.empty((target_height, target_width), dtype=bayer.dtype)
    resized[0::2, 0::2] = p00
    resized[0::2, 1::2] = p01
    resized[1::2, 1::2] = p11
    resized[1::2, 0::2] = p10
    return resized


def demosaic_menon2007(bayer: np.ndarray, pattern: str) -> np.ndarray:
    """
    High-quality demosaicing using colour-demosaicing DFPD (Menon 2007) method.
    Returns uint8 RGB image.
    """
    from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007

    bayer_float = bayer.astype(np.float32) / 255.0
    rgb = demosaicing_CFA_Bayer_Menon2007(bayer_float, pattern)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def demosaic_malvar_hqli(bayer: np.ndarray, pattern: str) -> np.ndarray:
    """
    HQLI - High-Quality Linear Interpolation (Malvar-He-Cutler) demosaicing.
    Returns uint8 RGB image.
    """
    from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004

    bayer_float = bayer.astype(np.float32) / 255.0
    rgb = demosaicing_CFA_Bayer_Malvar2004(bayer_float, pattern)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def demosaic_opencv_edge_aware(bayer: np.ndarray, pattern: str) -> np.ndarray:
    """
    Fallback demosaicing using OpenCV edge-aware Bayer conversion.
    Returns uint8 RGB image.
    """
    import cv2

    code_map = {
        "RGGB": cv2.COLOR_BayerRG2RGB_EA,
        "BGGR": cv2.COLOR_BayerBG2RGB_EA,
        "GRBG": cv2.COLOR_BayerGR2RGB_EA,
        "GBRG": cv2.COLOR_BayerGB2RGB_EA,
    }

    return cv2.cvtColor(bayer, code_map[pattern])


def convert_one(
    raw_path: Path,
    input_root: Path,
    output_root: Path,
    method: str,
    scale_down_bayer: bool,
    bayer_size: tuple[int, int],
) -> bool:
    match = RAW_RE.match(raw_path.name)
    if not match:
        return False

    width = int(match.group("w"))
    height = int(match.group("h"))
    pattern = match.group("pattern").upper()

    expected_size = width * height
    data = np.fromfile(raw_path, dtype=np.uint8)

    if data.size != expected_size:
        print(f"SKIP: {raw_path}")
        print(f"      expected {expected_size} bytes, got {data.size}")
        return False

    bayer = data.reshape((height, width))

    if scale_down_bayer:
        bayer = resize_bayer_planes(bayer, bayer_size)

    if method == "dfpd":
        rgb = demosaic_menon2007(bayer, pattern)
    elif method == "hqli":
        rgb = demosaic_malvar_hqli(bayer, pattern)
    elif method == "ea":
        rgb = demosaic_opencv_edge_aware(bayer, pattern)
    else:
        try:
            rgb = demosaic_menon2007(bayer, pattern)
        except Exception as e:
            print(f"DFPD failed for {raw_path}, fallback to OpenCV EA: {e}")
            rgb = demosaic_opencv_edge_aware(bayer, pattern)

    relative_path = raw_path.relative_to(input_root)
    out_path = output_root / relative_path.with_suffix(".jpg")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb, mode="RGB").save(
        out_path,
        format="JPEG",
        quality=85,
        optimize=True,
    )

    print(f"OK: {raw_path} -> {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Bayer RAW files to RGB JPEG using high-quality demosaicing."
    )
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument(
        "-m",
        "--method",
        choices=["auto", "dfpd", "hqli", "ea"],
        default="auto",
        help=(
            "Demosaicing method. Default: auto = DFPD with OpenCV EA fallback."
        ),
    )
    parser.add_argument(
        "-d",
        "--scale-down-bayer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize Bayer data before demosaicing. Default: enabled.",
    )
    parser.add_argument(
        "-s",
        "--bayer-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(384, 384),
        help="Size used when --scale-down-bayer is enabled. Default: 384 384.",
    )

    args = parser.parse_args()

    input_root = args.input_folder.resolve()
    output_root = args.output_folder.resolve()

    if not input_root.is_dir():
        raise RuntimeError(f"Input folder does not exist: {input_root}")

    total_raw = 0
    converted = 0

    for raw_path in input_root.rglob("*"):
        if not raw_path.is_file():
            continue

        if raw_path.suffix.lower() != ".raw":
            continue

        total_raw += 1

        if RAW_RE.match(raw_path.name):
            if convert_one(
                raw_path,
                input_root,
                output_root,
                args.method,
                args.scale_down_bayer,
                tuple(args.bayer_size),
            ):
                converted += 1
        else:
            print(f"SKIP name pattern mismatch: {raw_path}")

    print()
    print(f"RAW files found: {total_raw}")
    print(f"Converted:       {converted}")


if __name__ == "__main__":
    main()
