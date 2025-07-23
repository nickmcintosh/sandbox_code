"""Batch SuryaOCR over an images root directory.

Recursively walks a root images folder (e.g., project/data/images) and runs
Detection, Recognition, and Layout on every image file it finds. Outputs three
artifact types:

1. *Detections*  – bounding boxes for detected text regions.
2. *Recognition* – recognized text (per line / region) with confidences.
3. *Layout*      – higher-level layout grouping (blocks, reading order, etc.).

For convenience, a consolidated CSV is also emitted that flattens the per-image
results (image_path, region_id, text, confidence, bbox_x0, x1, y0, y1, block_id,
line_id, etc.) so you can quickly load results into Pandas / SQLite.

---
**Assumptions / Notes**

* You already have SuryaOCR installed and importable.
* The *exact* model-loading API in Surya has changed across versions. Below I
  provide two loader strategies:
  - **STRATEGY A (Predictor classes)** – preferred if you're on a recent Surya
    version exposing `DetectionPredictor`, `RecognitionPredictor`, and
    `LayoutPredictor` classes with `.from_pretrained(...)` constructors.
  - **STRATEGY B (legacy functional loaders)** – falls back to the style you
    sketched (segformer.load_model(), load_processor(), run_ocr(...)). If you're
    currently using that API, flip the `USE_LEGACY_API` constant to True and
    complete the TODOs with the correct imports you have in your environment.

* Languages: you mentioned French. Default below is `["fr", "en"]`. Adjust.
* Image formats: defaults to *.png, *.jpg, *.jpeg, *.tif, *.tiff, *.bmp, *.webp.
  Extend as needed.
* Memory: images are loaded one-at-a-time by default. If you want batching,
  increase `BATCH_SIZE`; note you'll need to ensure the loader functions accept
  batch lists. The safe default is 1.
* Error handling: we catch exceptions per image, log them, and continue.
* Output structure:

    output_root/
        detections/
            <relative_path_with_ext>.detections.json
        recognition/
            <relative_path_with_ext>.recognition.json
        layout/
            <relative_path_with_ext>.layout.json
        merged_results.csv
        run_manifest.json  # summary metadata of run

  The *relative_path_with_ext* uses the image path relative to the root, with
  path separators replaced by `__` so all files live in flat subdirs. Example:

      project/data/images/ANTI_OEDIPUS_FR/page_001.png
        → detections/ANTI_OEDIPUS_FR__page_001.png.detections.json

* Coordinates: Surya typically returns boxes in pixel coords (x0,y0,x1,y1) or
  polygon lists. We normalize to x0,y0,x1,y1 bounding boxes for CSV; polygon is
  preserved in JSON outputs if available.

---
USAGE EXAMPLES

Python module usage:

    python batch_surya_ocr_images.py \
        --images-root project/data/images \
        --output-root ocr/output/surya_batch_images \
        --languages fr en \
        --device auto

If you prefer *no CLI* and just edit constants in-file, set them below under the
`if __name__ == "__main__":` guard.

---
LIMITED GUARANTEE / FILL-IN REQUIRED

Because Surya's API surface has shifted, there are a few clearly marked TODOs.
Where I'm not 100% certain of the call signature, I both (a) include the likely
correct call for current releases, and (b) show you where to adapt for your
local version. The scaffolding (directory traversal, I/O, data structuring) is
solid; you'll mainly need to align the 3 predictor calls with your installed
Surya.

---
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image

# ---------------------------------------------------------------------------
# Surya imports – adjust to match your installed version
# ---------------------------------------------------------------------------
try:
    # Recent (preferred) predictor-class API
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor
    from surya.layout import LayoutPredictor
except ImportError:  # pragma: no cover - fallback for older versions
    DetectionPredictor = None  # type: ignore
    RecognitionPredictor = None  # type: ignore
    LayoutPredictor = None  # type: ignore

# Legacy functional API placeholders (adjust if you actually need them)
# from surya.detection import segformer   # example; comment/uncomment as needed
# from surya.recognition.model import load_model as load_rec_model
# from surya.recognition.processor import load_processor as load_rec_processor
# from surya.ocr import run_ocr  # etc.


# =============================================================================
# Configuration
# =============================================================================
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Flip True if you must use legacy loader calls; otherwise we'll use Predictor classes
USE_LEGACY_API = False

# If you know the exact model names / paths, set here (Predictor API)
DET_MODEL_NAME = None  # e.g., "surya/segformer_v1" or path to local weights
REC_MODEL_NAME = None  # e.g., "surya/recognition_v1"
LAY_MODEL_NAME = None  # e.g., "surya/layout_v1"

DEFAULT_LANGS = ["fr", "en"]


# =============================================================================
# Data containers
# =============================================================================
@dataclasses.dataclass
class LinePrediction:
    text: str
    confidence: Optional[float]
    bbox: Tuple[float, float, float, float]  # x0,y0,x1,y1
    polygon: Optional[List[Tuple[float, float]]] = None
    block_id: Optional[int] = None  # layout grouping id
    line_id: Optional[int] = None   # sequential within image

    def as_csv_row(self, image_rel: str) -> List[Union[str, float, int]]:
        return [
            image_rel,
            self.line_id if self.line_id is not None else "",
            self.block_id if self.block_id is not None else "",
            self.text,
            self.confidence if self.confidence is not None else "",
            *self.bbox,
        ]


# =============================================================================
# Utility: walk image tree
# =============================================================================

def iter_image_paths(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


# =============================================================================
# Model loading wrappers
# =============================================================================

def load_models(device: str = "auto"):
    """Load Surya detection, recognition, and layout models.

    Returns a tuple: (detector, recognizer, layout)
    """
    if not USE_LEGACY_API and DetectionPredictor is not None:
        det = DetectionPredictor.from_pretrained(DET_MODEL_NAME, device=device) if DET_MODEL_NAME else DetectionPredictor(device=device)
        rec = RecognitionPredictor.from_pretrained(REC_MODEL_NAME, device=device) if REC_MODEL_NAME else RecognitionPredictor(device=device)
        lay = LayoutPredictor.from_pretrained(LAY_MODEL_NAME, device=device) if LAY_MODEL_NAME else LayoutPredictor(device=device)
        return det, rec, lay

    # ------------------------------------------------------------------
    # Legacy path – ***You must fill in the correct calls for your version***
    # ------------------------------------------------------------------
    # Example sketch (replace with working imports / calls):
    # det_processor, det_model = segformer.load_processor(), segformer.load_model(device=device)
    # rec_model, rec_processor = load_rec_model(device=device), load_rec_processor()
    # lay_model, lay_processor = load_lay_model(device=device), load_lay_processor()
    # return (det_model, det_processor), (rec_model, rec_processor), (lay_model, lay_processor)
    raise RuntimeError("Legacy Surya API path not configured. Set USE_LEGACY_API and fill in loader calls.")


# =============================================================================
# Inference wrappers
# =============================================================================

def run_surya_predictors(
    image: Image.Image,
    detector: Any,
    recognizer: Any,
    layout_model: Any,
    languages: Sequence[str],
) -> List[LinePrediction]:
    """Unified inference wrapper for Predictor-class API.

    Returns list[LinePrediction].
    """
    # --- Detection -----------------------------------------------------
    det_preds = detector([image])  # expect list[DetResult]
    # Many Surya builds return polygons per line/word. We'll assume first element.
    det_result = det_preds[0]

    # det_result might expose .boxes (N,4) OR .polygons / .bboxes. We'll try to
    # access generically.
    if hasattr(det_result, "boxes"):
        det_boxes = det_result.boxes  # (N,4)
    elif hasattr(det_result, "bboxes"):
        det_boxes = det_result.bboxes
    else:
        # fallback from polygons -> boxes
        det_boxes = [polygon_to_bbox(poly) for poly in getattr(det_result, "polygons", [])]

    # --- Crop regions for recognition ---------------------------------
    crops: List[Image.Image] = []
    crop_polys: List[Any] = []
    for b in det_boxes:
        x0, y0, x1, y1 = map(int, b)
        crops.append(image.crop((x0, y0, x1, y1)))
        crop_polys.append(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))

    # --- Recognition ---------------------------------------------------
    rec_texts, rec_probs = [], []
    if crops:
        rec_out = recognizer(crops, languages=list(languages))  # expect list of strings or (text,conf)
        # Attempt to parse; Surya variants differ
        for item in rec_out:
            if isinstance(item, tuple) and len(item) == 2:
                txt, prob = item
            elif isinstance(item, dict):
                txt = item.get("text", "")
                prob = item.get("confidence") or item.get("prob")
            else:
                txt, prob = str(item), None
            rec_texts.append(txt)
            rec_probs.append(prob)
    else:
        rec_texts, rec_probs = [], []

    # --- Layout --------------------------------------------------------
    # Some LayoutPredictor variants expect the image and detection boxes; others
    # re-run detection internally. We'll attempt the common signature.
    try:
        lay_out = layout_model([image], [det_boxes])  # list[LayoutResult]
        lay_result = lay_out[0]
        # Suppose lay_result.blocks maps region indices -> block ids / reading order
        block_ids = getattr(lay_result, "block_ids", None)
        if block_ids is None:
            # Fallback: identity
            block_ids = list(range(len(det_boxes)))
    except Exception:  # pragma: no cover - fallback path
        block_ids = [None] * len(det_boxes)

    # --- Consolidate ---------------------------------------------------
    preds: List[LinePrediction] = []
    for i, box in enumerate(det_boxes):
        x0, y0, x1, y1 = box
        preds.append(
            LinePrediction(
                text=rec_texts[i] if i < len(rec_texts) else "",
                confidence=rec_probs[i] if i < len(rec_probs) else None,
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                polygon=list(map(tuple, crop_polys[i])) if i < len(crop_polys) else None,
                block_id=block_ids[i] if i < len(block_ids) else None,
                line_id=i,
            )
        )
    return preds


# =============================================================================
# Polygon → bbox helper
# =============================================================================

def polygon_to_bbox(poly: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


# =============================================================================
# Serialization helpers
# =============================================================================

def rel_to_safe(rel_path: Path) -> str:
    """Convert a relative path into a filesystem-safe flat token."""
    return "__".join(rel_path.parts)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =============================================================================
# Core processing
# =============================================================================

def process_image(
    image_path: Path,
    root: Path,
    detector: Any,
    recognizer: Any,
    layout_model: Any,
    languages: Sequence[str],
    output_root: Path,
) -> Optional[List[LinePrediction]]:
    """Run Surya models on a single image and write per-image JSON artifacts."""
    rel = image_path.relative_to(root)
    safe = rel_to_safe(rel)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:  # pragma: no cover
        logging.error("Failed to open %s: %s", image_path, e)
        return None

    try:
        preds = run_surya_predictors(image, detector, recognizer, layout_model, languages=languages)
    except Exception as e:  # pragma: no cover
        logging.exception("OCR failed for %s", image_path)
        return None

    # Serialize detailed outputs ---------------------------------------
    det_payload = [{"bbox": p.bbox, "polygon": p.polygon, "confidence": p.confidence} for p in preds]
    rec_payload = [{"text": p.text, "confidence": p.confidence, "line_id": p.line_id} for p in preds]
    lay_payload = [{"line_id": p.line_id, "block_id": p.block_id} for p in preds]

    write_json(det_payload, output_root / "detections" / f"{safe}.detections.json")
    write_json(rec_payload, output_root / "recognition" / f"{safe}.recognition.json")
    write_json(lay_payload, output_root / "layout" / f"{safe}.layout.json")

    return preds


# =============================================================================
# Batch driver
# =============================================================================

def run_batch(
    images_root: Union[str, Path],
    output_root: Union[str, Path],
    languages: Sequence[str] = DEFAULT_LANGS,
    device: str = "auto",
) -> Path:
    images_root = Path(images_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info("Loading Surya models (device=%s)...", device)
    det, rec, lay = load_models(device=device)

    logging.info("Scanning for images under %s ...", images_root)
    image_paths = sorted(iter_image_paths(images_root))
    logging.info("Found %d image(s).", len(image_paths))

    merged_csv_path = output_root / "merged_results.csv"
    with merged_csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image_rel", "line_id", "block_id", "text", "confidence", "x0", "y0", "x1", "y1"])

        for img_p in image_paths:
            preds = process_image(img_p, images_root, det, rec, lay, languages, output_root)
            if not preds:
                continue
            rel = img_p.relative_to(images_root).as_posix()
            for p in preds:
                writer.writerow(p.as_csv_row(rel))

    # Manifest ---------------------------------------------------------
    manifest = {
        "images_root": str(images_root),
        "output_root": str(output_root),
        "languages": list(languages),
        "num_images": len(image_paths),
    }
    write_json(manifest, output_root / "run_manifest.json")

    logging.info("Batch complete. Results in %s", output_root)
    return output_root


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch SuryaOCR over an images directory.")
    p.add_argument("--images-root", required=False, default="project/data/images", help="Root directory containing images (recursively processed).")
    p.add_argument("--output-root", required=False, default="ocr/output/surya_batch_images", help="Directory to write OCR outputs.")
    p.add_argument("--languages", nargs="*", default=DEFAULT_LANGS, help="Language codes for recognition (space-separated).")
    p.add_argument("--device", default="auto", help="Device string passed to Surya models (e.g., 'cpu', 'cuda', 'mps', 'auto').")
    p.add_argument("--legacy", action="store_true", help="Use legacy Surya API loaders.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    ap = build_arg_parser()
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelsname)s %(message)s")

    run_batch(
        images_root=args.images_root,
        output_root=args.output_root,
        languages=args.languages,
        device=args.device,
    )
