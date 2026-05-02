# re-bayer

Utility for converting RGB images into RAW Bayer files.

## Features

- Scans an input folder recursively for image files
- Supports `.jpeg`, `.jpg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`
- Writes output files as `source_filename_WIDTHxHEIGHT@PATTERN.RAW`
- Preserves the input folder structure under the output folder
- Supports Bayer patterns `RGGB`, `GRBG`, `GBRG`, `BGGR`
- Views de-Bayered `.RAW` files with OpenCV, using the filename suffix by default
- Implements algorithm `0`:
  for each RAW pixel, takes the matching `R`, `G`, or `B` value from the same RGB pixel according to the selected Bayer pattern
- Includes a validation procedure that de-Bayers RAW output with OpenCV, computes PSNR against the source image, and writes a CSV report

Algorithms `1`, `2`, and later are reserved for future work and currently return `NotImplementedError`.

## Usage

```bash
./.venv/bin/python re-bayer.py
```

```bash
./.venv/bin/python re-bayer.py --input-folder ./images --output-folder ./our --pattern RGGB
```

For example, `./images/class_a/frame.jpg` is written as
`./our/class_a/frame_WIDTHxHEIGHT@RGGB.RAW`.

```bash
./.venv/bin/python validate.py --input-folder ./images --output-folder ./our --pattern RGGB --min-psnr 20.0 --report-csv ./validation_report.csv
```

```bash
./.venv/bin/python view_raw.py ./our/source_filename_500x375@RGGB.RAW
```

```bash
./.venv/bin/python view_raw.py ./frame.RAW --width 500 --height 375 --bayer-start RGGB
```

## Arguments

- `--input-folder`, `-i`
  Input folder with source image files, scanned recursively. Default: `./`
- `--output-folder`, `-o`
  Output folder for `.RAW` files. Relative input subfolders are preserved. Default: `./our`
- `--pattern`, `-p`
  Bayer pattern: `RGGB`, `GRBG`, `GBRG`, `BGGR`. Default: `RGGB`

## Validation Procedure

Use `validate.py` after generating `.RAW` files.

What it does:

- Finds the expected RAW file for each source image using the format `source_filename_WIDTHxHEIGHT@PATTERN.RAW`
- Preserves the source image's relative folder path when looking under the output folder
- De-Bayers the RAW file with OpenCV using the selected Bayer pattern
- Calculates PSNR against the original source image
- Prints an error message when PSNR is below the configured threshold
- Stores `input_file`, `output_file`, `status`, `error_code`, `psnr_db`, and `message` in a CSV report

Validation arguments:

- `--input-folder`, `-i`
  Folder with source image files, scanned recursively. Default: `./`
- `--output-folder`, `-o`
  Folder with output `.RAW` files. Relative input subfolders are preserved. Default: `./our`
- `--pattern`, `-p`
  Bayer pattern: `RGGB`, `GRBG`, `GBRG`, `BGGR`. Default: `RGGB`
- `--min-psnr`
  Minimum acceptable PSNR in dB. Default: `20.0`
- `--report-csv`
  CSV report path. Default: `./validation_report.csv`

Error codes:

- `0`: validation passed
- `10`: RAW output file missing
- `11`: RAW size does not match source image dimensions
- `12`: de-Bayered image size does not match source image size
- `13`: PSNR below threshold
- `14`: other processing failure

## RAW Viewer

Use `view_raw.py` to open a de-Bayered preview of an 8-bit `.RAW` file.

By default, metadata is parsed from filenames that end with `WIDTHxHEIGHT@PATTERN`,
for example `ILSVRC2012_val_00000001_500x375@RGGB.RAW`.

Viewer arguments:

- `raw_file`
  Path to the `.RAW` file to view
- `--width`
  Override image width
- `--height`
  Override image height
- `--bayer-start`
  Override Bayer start/pattern: `RGGB`, `GRBG`, `GBRG`, `BGGR`
- `--wait-ms`
  Milliseconds to keep the OpenCV window open. Default: `0`, wait for a key press
