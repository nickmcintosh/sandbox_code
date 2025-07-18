#!/usr/bin/env python3
import multiprocessing as mp

if __name__ == "__main__":
    # ─── 1) Use spawn to avoid fork‐related crashes ───────────────────────────
    mp.set_start_method("spawn", force=True)

    # ─── 2) Force Surya (and Torch) onto CPU ────────────────────────────────
    import os
    os.environ["TORCH_DEVICE"] = "cpu"       # Surya picks up CPU device
    os.environ["OMP_NUM_THREADS"] = "1"      # limit OpenMP parallelism
    os.environ["MKL_NUM_THREADS"] = "1"      # limit MKL parallelism

    # ─── 3) Imports ──────────────────────────────────────────────────────────
    import json, csv, gc
    import torch
    from pdf2image import pdfinfo_from_path, convert_from_path  # pip install pdf2image
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor
    from surya.layout    import LayoutPredictor

    # ─── 4) Cap Torch threading ─────────────────────────────────────────────
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # ─── 5) Paths & prep ────────────────────────────────────────────────────
    INPUT_PDF  = "ocr/input/PDF/KAFKA_FR-5-9.pdf"
    OUTPUT_DIR = "ocr/output/surya_KAFKA_FR-5-9"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─── 6) Figure out how many pages we have ───────────────────────────────
    info       = pdfinfo_from_path(INPUT_PDF)
    num_pages  = info["Pages"]

    # ─── 7) Load Surya models ────────────────────────────────────────────────
    detector   = DetectionPredictor()
    recognizer = RecognitionPredictor()
    layouter   = LayoutPredictor()

    # ─── 8) Process one page at a time ───────────────────────────────────────
    raw_ocr    = []
    raw_layout = []

    for p in range(1, num_pages + 1):
        # convert only page `p` at 150 DPI, single‐threaded
        images = convert_from_path(
            INPUT_PDF, dpi=150,
            first_page=p, last_page=p,
            thread_count=1
        )
        img = images[0]

        # run detection+OCR and layout
        raw_ocr.append   (recognizer([img], det_predictor=detector)[0])
        raw_layout.append(layouter([img])[0])

        # free memory before next page
        del img, images
        gc.collect()

    # ─── 9) Helper to turn Surya results into plain dicts ────────────────────
    def to_dict(obj):
        if hasattr(obj, "dict"):       # e.g. pydantic models
            return {k: to_dict(v) for k, v in obj.dict().items()}
        if hasattr(obj, "__dict__"):    # dataclasses or simple classes
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [to_dict(v) for v in obj]
        if isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj  # primitives (str, int, float, etc.)

    ocr_results    = to_dict(raw_ocr)
    layout_results = to_dict(raw_layout)

    # ─── 10) Write JSON ──────────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "ocr_results.json"), "w", encoding="utf-8") as jf:
        json.dump(ocr_results, jf, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "layout_results.json"), "w", encoding="utf-8") as jf:
        json.dump(layout_results, jf, ensure_ascii=False, indent=2)

    # ─── 11) Write a simple CSV of text‐lines ────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "ocr_lines.csv"), "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["page", "text", "confidence", "bbox"])
        for page in ocr_results:  # now a list of dicts
            pnum = page.get("page")
            for line in page["text_lines"]:
                bbox_str = ",".join(map(str, line["bbox"]))
                writer.writerow([pnum, line["text"], line["confidence"], bbox_str])

    print("Wrote results to", OUTPUT_DIR)
