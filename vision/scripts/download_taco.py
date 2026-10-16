"""Download TACO from Zenodo with aria2c and unpack it straight into a YOLO dataset.

Steps:
  1. Download TACO.zip with aria2c (resumable, MD5-verified).
  2. Extract the images from the zip directly into taco-yolo/images/{train,val}/,
     plus annotations.json (TACO's original COCO labels).
  3. Write YOLO labels and data.yaml from annotations.json.
  4. Delete TACO.zip (only with --delete-zip; kept by default).

Safe to re-run: once taco-yolo/ exists, steps 1, 2 and 4 are skipped and only the
labels are rewritten, so changing --classes, --val-frac or --seed takes seconds
and doesn't need the zip.

Usage (from the vision/ directory):
    python scripts/download_taco.py                  # 28 supercategories (default)
    python scripts/download_taco.py --classes all    # all 60 TACO categories
    python scripts/download_taco.py --classes trash  # single "trash" class
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/3587843/files/TACO.zip?download=1"
MD5 = "e9149407d883e21a8d224feef8210920"
ZIP_NAME = "TACO.zip"
DATA_PREFIX = "TACO/data/"  # the only part of the zip we use

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
ZIP_PATH = DATASETS_DIR / ZIP_NAME
YOLO_DIR = DATASETS_DIR / "taco-yolo"
ANNOTATIONS = YOLO_DIR / "annotations.json"
SPLITS = ("train", "val")


def is_built():
    return ANNOTATIONS.exists()


def download():
    if shutil.which("aria2c") is None:
        sys.exit(
            "aria2c not found on PATH. Install it:\n"
            "  Windows: winget install aria2.aria2\n"
            "  macOS:   brew install aria2\n"
            "  Linux:   sudo apt install aria2  (or your distro's package manager)"
        )

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c",
        URL,
        "--dir", str(DATASETS_DIR),
        "--out", ZIP_NAME,
        "--continue=true",              # resume a partial download
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--max-tries=0",                # retry forever on network errors
        "--retry-wait=10",
        "--checksum", f"md5={MD5}",     # verify (and re-check an existing file)
        "--check-integrity=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--file-allocation=none",
        "--summary-interval=30",
    ]
    subprocess.run(cmd, check=True)


def yolo_name(file_name):
    """batch_1/000006.jpg -> batch_1_000006.jpg (names repeat across batches)."""
    return file_name.replace("/", "_").lower()


def split_images(images, val_frac, seed):
    """Deterministically assign each image to train or val."""
    images = sorted(images, key=lambda im: im["id"])
    random.Random(seed).shuffle(images)
    n_val = round(len(images) * val_frac)
    return {im["id"]: ("val" if i < n_val else "train") for i, im in enumerate(images)}


def extract(val_frac, seed):
    """Extract annotations.json and the images from the zip straight into taco-yolo/.

    The zip is a snapshot of the TACO repo, so everything outside TACO/data/
    (.git, old detector code, __MACOSX) is ignored. Extraction goes to a temp
    dir first so an interrupted run never looks finished.
    """
    tmp_dir = DATASETS_DIR / "taco-yolo.partial"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)  # leftover from an older version of this script

    print(f"Extracting {ZIP_PATH} -> {YOLO_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        coco_bytes = zf.read(DATA_PREFIX + "annotations.json")
        coco = json.loads(coco_bytes)
        split_of = split_images(coco["images"], val_frac, seed)

        for split in SPLITS:
            (tmp_dir / "images" / split).mkdir(parents=True)
        for im in coco["images"]:
            dest = tmp_dir / "images" / split_of[im["id"]] / yolo_name(im["file_name"])
            with zf.open(DATA_PREFIX + im["file_name"]) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    (tmp_dir / "annotations.json").write_bytes(coco_bytes)
    tmp_dir.rename(YOLO_DIR)
    print(f"Extracted {len(coco['images'])} images")


def class_mapping(categories, mode):
    """Return (names, {coco_category_id: yolo_class_id}) for the chosen class mode."""
    if mode == "trash":
        return ["trash"], {c["id"]: 0 for c in categories}
    if mode == "all":
        cats = sorted(categories, key=lambda c: c["id"])
        return [c["name"] for c in cats], {c["id"]: i for i, c in enumerate(cats)}
    # "super": group the 60 categories into TACO's 28 supercategories
    names = list(dict.fromkeys(c["supercategory"] for c in sorted(categories, key=lambda c: c["id"])))
    return names, {c["id"]: names.index(c["supercategory"]) for c in categories}


def write_labels(class_mode, val_frac, seed):
    """Write YOLO label files and data.yaml from taco-yolo/annotations.json.

    Each label line is "<class> <x_center> <y_center> <width> <height>", all 0-1.
    Images are moved between train/ and val/ if the split settings changed.
    """
    coco = json.loads(ANNOTATIONS.read_text())
    names, cat_to_class = class_mapping(coco["categories"], class_mode)
    split_of = split_images(coco["images"], val_frac, seed)

    boxes_by_image = {}
    for ann in coco["annotations"]:
        boxes_by_image.setdefault(ann["image_id"], []).append(ann)

    labels_dir = YOLO_DIR / "labels"
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    for split in SPLITS:
        (labels_dir / split).mkdir(parents=True)

    for im in coco["images"]:
        name = yolo_name(im["file_name"])
        split = split_of[im["id"]]
        other = "val" if split == "train" else "train"
        wrong_place = YOLO_DIR / "images" / other / name
        if wrong_place.exists():
            wrong_place.rename(YOLO_DIR / "images" / split / name)

        # Width/height in annotations.json already account for EXIF rotation,
        # matching how Ultralytics loads the images.
        w, h = im["width"], im["height"]
        lines = []
        for ann in boxes_by_image.get(im["id"], []):
            x, y, bw, bh = ann["bbox"]  # COCO: top-left corner + size, in pixels
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + bw, w), min(y + bh, h)
            if x2 <= x1 or y2 <= y1:
                continue
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            lines.append(f"{cat_to_class[ann['category_id']]} {xc:.6f} {yc:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        label = labels_dir / split / (Path(name).stem + ".txt")
        label.write_text("\n".join(lines) + ("\n" if lines else ""))

    yaml_lines = [
        f"path: {YOLO_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
        *(f"  {i}: {json.dumps(name)}" for i, name in enumerate(names)),
    ]
    (YOLO_DIR / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    counts = {s: sum(1 for v in split_of.values() if v == s) for s in SPLITS}
    print(f"Labels written ({class_mode}: {len(names)} classes): "
          f"{counts['train']} train / {counts['val']} val -> {YOLO_DIR / 'data.yaml'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", choices=["super", "all", "trash"], default="super",
                        help="super: 28 TACO supercategories (default), all: 60 categories, trash: 1 class")
    parser.add_argument("--val-frac", type=float, default=0.2, help="fraction of images for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the train/val split (default 42)")
    parser.add_argument("--delete-zip", action="store_true", help="delete TACO.zip after extracting (kept by default)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if is_built():
        print(f"Already extracted: {YOLO_DIR}")
    else:
        download()
        extract(args.val_frac, args.seed)
        if args.delete_zip:
            ZIP_PATH.unlink()
            print(f"Deleted {ZIP_PATH}")
    write_labels(args.classes, args.val_frac, args.seed)
