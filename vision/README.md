# IZZ-E Vision

Trash detection for IZZ-E using YOLO fine-tuned on the [TACO](http://tacodataset.org/) dataset.

## Setup

Create and activate a venv, then install dependencies.

```sh
python -m venv .venv
```

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

```sh
pip install -r requirements.txt
```

## Dataset

Download [TACO](https://zenodo.org/records/3587843) into `datasets/` (requires [aria2c](https://aria2.github.io/): `winget install aria2.aria2` on Windows, `brew install aria2` on macOS, `sudo apt install aria2` on Debian/Ubuntu):

```sh
python scripts/download_taco.py
```

The script resumes interrupted downloads, verifies the MD5 checksum, extracts the images straight into a YOLO dataset at `datasets/taco-yolo/` (80/20 train/val split), and keeps the zip (`--delete-zip` to remove it and save 2.5 GB). It's safe to re-run: once `taco-yolo/` exists, only the labels are rewritten, so switching `--classes` takes seconds and doesn't re-download anything.

By default the 60 TACO categories are grouped into its 28 supercategories. Use `--classes all` for all 60, or `--classes trash` for a single class. Run with `--help` for all options.
