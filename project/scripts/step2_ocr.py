### scripts/step2_ocr.py
# from pathlib import Path
# import os, json, gc, yaml
# import torch
# from PIL import Image
# from surya.detection   import DetectionPredictor
# from surya.recognition import RecognitionPredictor
# from surya.layout      import LayoutPredictor

# # load config
# dir_here = Path(__file__).parent.parent
# SCRIPT_ROOT = Path(__file__).resolve().parent.parent
# CONFIG_PATH = SCRIPT_ROOT / "config.yaml"

# cfg = yaml.safe_load(CONFIG_PATH.read_text())
# IMAGES_DIR = Path(cfg["image_dir"])
# OCR_DIR    = Path(cfg["ocr_dir"])

# # ensure CPU usage and limit threads
# os.environ["TORCH_DEVICE"] = "cpu"
# torch.set_num_threads(cfg.get("torch_threads", 1))
# torch.set_num_interop_threads(cfg.get("torch_threads", 1))

# # Initialize models once
# detector   = DetectionPredictor()
# recognizer = RecognitionPredictor()
# layouter   = LayoutPredictor()


# def run_ocr():
#     """
#     For each document subdir under images/, process PNGs page-by-page
#     to produce ocr_results.json & layout_results.json under ocr/.
#     """
#     OCR_DIR.mkdir(parents=True, exist_ok=True)

#     for doc_dir in sorted(IMAGES_DIR.iterdir()):
#         if not doc_dir.is_dir():
#             continue
#         name = doc_dir.name
#         out_dir = OCR_DIR / name
#         out_dir.mkdir(parents=True, exist_ok=True)

#         # gather page images
#         pages = []
#         for img_path in sorted(doc_dir.glob("*.png")):
#             pages.append(Image.open(img_path))

#         ocr_results = []
#         layout_results = []
#         for img in pages:
#             res_ocr = recognizer([img], det_predictor=detector)[0]
#             res_lay = layouter([img])[0]
#             ocr_results.append(res_ocr)
#             layout_results.append(res_lay)
#             img.close()
#             gc.collect()

#         # serialize
#         with open(out_dir / "ocr_results.json", "w", encoding="utf-8") as jf:
#             json.dump([r.dict() if hasattr(r, 'dict') else r.__dict__ for r in ocr_results], jf, ensure_ascii=False, indent=2)
#         with open(out_dir / "layout_results.json", "w", encoding="utf-8") as jf:
#             json.dump([r.dict() if hasattr(r, 'dict') else r.__dict__ for r in layout_results], jf, ensure_ascii=False, indent=2)

# scripts/step2_ocr.py
#!/usr/bin/env python3
import argparse
import json
import gc
import os
import yaml
import torch
from pathlib import Path
from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.layout import LayoutPredictor

# ─── CLI Args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Step2 OCR: process a doc or page range, with CPU/MPS fallback & downsampling"
)
parser.add_argument("--doc_name",   required=True, help="Document subfolder to process")
parser.add_argument("--cpu_only",   action="store_true", help="Force CPU mode")
parser.add_argument("--downscale",  type=float,   default=1.0, help="Resize factor (e.g. 0.5)")
parser.add_argument("--start_page", type=int,     default=1,   help="1-based first page to process")
parser.add_argument("--end_page",   type=int,     default=None, help="1-based last page to process")
args = parser.parse_args()

# Optionally force CPU backend before importing Surya drivers
if args.cpu_only:
    os.environ["TORCH_DEVICE"] = "cpu"

# ─── Config & Paths ─────────────────────────────────────────────────────────
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((SCRIPT_ROOT / "config.yaml").read_text())
IMAGES_DIR = SCRIPT_ROOT / cfg["image_dir"]
OCR_DIR    = SCRIPT_ROOT / cfg["ocr_dir"]

# ─── Device Selection ───────────────────────────────────────────────────────
def get_device(cpu_only: bool):
    if cpu_only:
        return torch.device("cpu")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ─── Core Batch Runner ─────────────────────────────────────────────────────
def run_batch(name: str,
              device: torch.device,
              downscale: float,
              start_page: int,
              end_page: int):
    img_dir = IMAGES_DIR / name
    all_imgs = sorted(img_dir.glob("*.png"))
    total = len(all_imgs)
    if end_page is None or end_page > total:
        end_page = total
    start_idx = max(1, start_page)
    end_idx = max(start_idx, end_page)
    batch_imgs = all_imgs[start_idx-1:end_idx]

    out_dir = OCR_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ocr_file    = out_dir / "ocr_results.jsonl"
    layout_file = out_dir / "layout_results.jsonl"

    # clear on first slice
    if start_idx == 1:
        if ocr_file.exists(): ocr_file.unlink()
        if layout_file.exists(): layout_file.unlink()

    print(f"[step2] 📖 '{name}' pages {start_idx}–{end_idx} on {device}")

    # load models
    detector   = DetectionPredictor()
    recognizer = RecognitionPredictor()
    layouter   = LayoutPredictor()
    for m in (detector.model, recognizer.model, layouter.model):
        m.to(device)
        if cfg.get("half_precision", False):
            m.half()

    idx = start_idx
    for img_path in batch_imgs:
        print(f"[step2]    Page {idx}: {img_path.name}")

        # Verify image integrity
        try:
            with Image.open(img_path) as tmp:
                tmp.verify()
        except Exception as e:
            print(f"[step2] ⚠️ Corrupt or unreadable image {img_path.name}: {e}. Skipping.")
            idx += 1
            continue

        # Re-open image after verify
        with Image.open(img_path).convert("RGB") as img:
            if downscale != 1.0:
                img = img.resize((int(img.width * downscale), int(img.height * downscale)))
            with torch.inference_mode():
                ocr_res = recognizer([img], det_predictor=detector)[0]
                lay_res = layouter([img])[0]

        # Enrich results with page metadata
        ocr_dict = ocr_res.dict()
        layout_dict = lay_res.dict()
        ocr_dict["page"] = idx
        ocr_dict["image_name"] = img_path.name
        layout_dict["page"] = idx
        layout_dict["image_name"] = img_path.name

        # Append enriched records
        with open(ocr_file,    "a", encoding="utf-8") as f:
            f.write(json.dumps(ocr_dict) + "\n")
        with open(layout_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(layout_dict) + "\n")

        idx += 1
        del ocr_res, lay_res
        gc.collect()

    # teardown
    del detector, recognizer, layouter
    gc.collect()
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    print(f"[step2] ✅ Done pages {start_idx}–{end_idx}")

# ─── Script Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device(args.cpu_only)
    run_batch(
        name       = args.doc_name,
        device     = device,
        downscale  = args.downscale,
        start_page = args.start_page,
        end_page   = args.end_page
    )