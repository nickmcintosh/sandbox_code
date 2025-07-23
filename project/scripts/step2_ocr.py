# # scripts/step2_ocr.py
# #!/usr/bin/env python3
# import argparse
# import json
# import gc
# import os
# import yaml
# import torch
# from pathlib import Path
# from PIL import Image
# from surya.detection import DetectionPredictor
# from surya.recognition import RecognitionPredictor
# from surya.layout import LayoutPredictor

# # ─── CLI Args ───────────────────────────────────────────────────────────────
# parser = argparse.ArgumentParser(
#     description="Step2 OCR: process a doc or page range, with CPU/MPS fallback & downsampling"
# )
# parser.add_argument("--doc_name",   required=True, help="Document subfolder to process")
# parser.add_argument("--cpu_only",   action="store_true", help="Force CPU mode")
# parser.add_argument("--downscale",  type=float,   default=1.0, help="Resize factor (e.g. 0.5)")
# parser.add_argument("--start_page", type=int,     default=1,   help="1-based first page to process")
# parser.add_argument("--end_page",   type=int,     default=None, help="1-based last page to process")
# args = parser.parse_args()

# # Optionally force CPU backend before importing Surya drivers
# if args.cpu_only:
#     os.environ["TORCH_DEVICE"] = "cpu"

# # ─── Config & Paths ─────────────────────────────────────────────────────────
# SCRIPT_ROOT = Path(__file__).resolve().parents[1]
# cfg = yaml.safe_load((SCRIPT_ROOT / "config.yaml").read_text())
# IMAGES_DIR = SCRIPT_ROOT / cfg["image_dir"]
# OCR_DIR    = SCRIPT_ROOT / cfg["ocr_dir"]

# # ─── Device Selection ───────────────────────────────────────────────────────
# def get_device(cpu_only: bool):
#     if cpu_only:
#         return torch.device("cpu")
#     return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# # ─── Core Batch Runner ─────────────────────────────────────────────────────
# def run_batch(name: str,
#               device: torch.device,
#               downscale: float,
#               start_page: int,
#               end_page: int):
#     img_dir = IMAGES_DIR / name
#     all_imgs = sorted(img_dir.glob("*.png"))
#     total = len(all_imgs)
#     if end_page is None or end_page > total:
#         end_page = total
#     start_idx = max(1, start_page)
#     end_idx = max(start_idx, end_page)
#     batch_imgs = all_imgs[start_idx-1:end_idx]

#     out_dir = OCR_DIR / name
#     out_dir.mkdir(parents=True, exist_ok=True)
#     ocr_file    = out_dir / "ocr_results.jsonl"
#     layout_file = out_dir / "layout_results.jsonl"

#     # clear on first slice
#     if start_idx == 1:
#         if ocr_file.exists(): ocr_file.unlink()
#         if layout_file.exists(): layout_file.unlink()

#     print(f"[step2] 📖 '{name}' pages {start_idx}–{end_idx} on {device}")

#     # load models
#     detector   = DetectionPredictor()
#     recognizer = RecognitionPredictor()
#     layouter   = LayoutPredictor()
#     for m in (detector.model, recognizer.model, layouter.model):
#         m.to(device)
#         if cfg.get("half_precision", False):
#             m.half()

#     idx = start_idx
#     for img_path in batch_imgs:
#         print(f"[step2]    Page {idx}: {img_path.name}")

#         # Verify image integrity
#         try:
#             with Image.open(img_path) as tmp:
#                 tmp.verify()
#         except Exception as e:
#             print(f"[step2] ⚠️ Corrupt or unreadable image {img_path.name}: {e}. Skipping.")
#             idx += 1
#             continue

#         # Re-open image after verify
#         with Image.open(img_path).convert("RGB") as img:
#             if downscale != 1.0:
#                 img = img.resize((int(img.width * downscale), int(img.height * downscale)))
#             with torch.inference_mode():
#                 ocr_res = recognizer([img], det_predictor=detector)[0]
#                 lay_res = layouter([img])[0]

#         # Enrich results with page metadata
#         ocr_dict = ocr_res.dict()
#         layout_dict = lay_res.dict()
#         ocr_dict["page"] = idx
#         ocr_dict["image_name"] = img_path.name
#         layout_dict["page"] = idx
#         layout_dict["image_name"] = img_path.name

#         # Append enriched records
#         with open(ocr_file,    "a", encoding="utf-8") as f:
#             f.write(json.dumps(ocr_dict) + "\n")
#         with open(layout_file, "a", encoding="utf-8") as f:
#             f.write(json.dumps(layout_dict) + "\n")

#         idx += 1
#         del ocr_res, lay_res
#         gc.collect()

#     # teardown
#     del detector, recognizer, layouter
#     gc.collect()
#     if device.type == "mps":
#         try:
#             torch.mps.empty_cache()
#         except Exception:
#             pass

#     print(f"[step2] ✅ Done pages {start_idx}–{end_idx}")

# # ─── Script Entry Point ────────────────────────────────────────────────────
# if __name__ == "__main__":
#     device = get_device(args.cpu_only)
#     run_batch(
#         name       = args.doc_name,
#         device     = device,
#         downscale  = args.downscale,
#         start_page = args.start_page,
#         end_page   = args.end_page
#     )

#!/usr/bin/env python3
"""
Step2 OCR: batch inference with robust resource management on macOS M2.
This version avoids multiprocessing semaphores entirely by not using torch.multiprocessing.
"""
import torch
# Pin CPU threads to reduce OpenMP overhead
torch.set_num_threads(1)

import json
import gc
import yaml
from pathlib import Path
from PIL import Image

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.layout import LayoutPredictor

# ─── Configuration ───────────────────────────────────────────────────────────
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((SCRIPT_ROOT / 'config.yaml').read_text())
IMAGES_DIR = SCRIPT_ROOT / cfg['image_dir']
OCR_DIR    = SCRIPT_ROOT / cfg['ocr_dir']

# ─── Device Selection ─────────────────────────────────────────────────────────
def get_device(cpu_only: bool):
    """Pick CPU or MPS (Metal) device for inference."""
    if cpu_only:
        return torch.device('cpu')
    return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# ─── Core Batch Runner ────────────────────────────────────────────────────────
def run_batch(name: str,
              device: torch.device,
              downscale: float,
              start_page: int,
              end_page: int):
    """
    Run OCR and layout extraction for a specified page range.
    Executes one-shot batch inference, cleans GPU cache, and tears down models explicitly.
    """
    img_dir = IMAGES_DIR / name
    all_imgs = sorted(img_dir.glob('*.png'))
    total = len(all_imgs)

    # Determine slice boundaries\    
    if end_page is None or end_page > total:
        end_page = total
    start_idx = max(1, start_page)
    end_idx   = max(start_idx, end_page)
    slice_paths = all_imgs[start_idx - 1:end_idx]

    out_dir = OCR_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ocr_file    = out_dir / 'ocr_results.jsonl'
    layout_file = out_dir / 'layout_results.jsonl'

    # Clear on first slice
    if start_idx == 1:
        ocr_file.unlink(missing_ok=True)
        layout_file.unlink(missing_ok=True)

    print(f"[step2] '{name}' pages {start_idx}–{end_idx} on {device}")

    # Load models once per batch
    detector   = DetectionPredictor(); recognizer = RecognitionPredictor(); layouter = LayoutPredictor()
    for model in (detector.model, recognizer.model, layouter.model):
        model.to(device)
        if cfg.get('half_precision', False):
            model.half()

    # Prepare images list
    images, pages, names = [], [], []
    idx = start_idx
    for img_path in slice_paths:
        try:
            with Image.open(img_path) as tmp:
                tmp.verify()
        except Exception as e:
            print(f"[step2] ⚠️ Corrupt {img_path.name}: {e}. Skipping.")
            idx += 1
            continue

        img = Image.open(img_path).convert('RGB')
        # Downscale and enforce multiple-of-64 dimensions
        w, h = int(img.width * downscale), int(img.height * downscale)
        w = ((w + 63) // 64) * 64; h = ((h + 63) // 64) * 64
        img = img.resize((w, h))
        images.append(img); pages.append(idx); names.append(img_path.name)
        idx += 1

    # One-shot inference
    if images:
        with torch.inference_mode():
            ocr_results    = recognizer(images,    det_predictor=detector)
            layout_results = layouter(images)

        # Write JSONL
        # Write JSONL
        with open(ocr_file, 'a', encoding='utf-8') as f_ocr, \
            open(layout_file, 'a', encoding='utf-8') as f_layout:
            for ocr_res, lay_res, pg, nm, img in zip(ocr_results, layout_results, pages, names, images):
                rec = ocr_res.dict(); rec.update({'page': pg, 'image_name': nm}); f_ocr.write(json.dumps(rec) + '\n')
                rec = lay_res.dict(); rec.update({'page': pg, 'image_name': nm}); f_layout.write(json.dumps(rec) + '\n')
                img.close()

# Clear GPU cache


        # Clear GPU cache\        
        if device.type == 'mps':
            try: torch.mps.empty_cache()
            except: pass

    # Explicit teardown
    del detector, recognizer, layouter, images, pages, names
    gc.collect()
    print(f"[step2] ✅ Done pages {start_idx}–{end_idx}")

# ─── Script Entry Point ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Step2 OCR CLI')
    parser.add_argument('--doc_name', required=True)
    parser.add_argument('--cpu_only', action='store_true')
    parser.add_argument('--downscale', type=float, default=1.0)
    parser.add_argument('--start_page', type=int, default=1)
    parser.add_argument('--end_page', type=int, default=None)
    args = parser.parse_args()
    device = get_device(args.cpu_only)
    run_batch(args.doc_name, device, args.downscale, args.start_page, args.end_page)
