#!/usr/bin/env python3
import json
import csv
import os
import sys

# ─── 1) Paths to your existing JSON outputs ───────────────────────────────
OCR_JSON    = "ocr/output/surya_KAFKA_FR-5-9/ocr_results.json"
LAYOUT_JSON = "ocr/output/surya_KAFKA_FR-5-9/layout_results.json"
CSV_OUTPUT  = "ocr/output/surya_KAFKA_FR-5-9/blocks.csv"

# ─── 2) Helper: test whether the center of a text‐line bbox falls in a block bbox ─
def line_in_block(line_bbox, block_bbox):
    lx1, ly1, lx2, ly2 = line_bbox
    bx1, by1, bx2, by2 = block_bbox
    cx, cy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
    return bx1 <= cx <= bx2 and by1 <= cy <= by2

# ─── 3) Load a JSON file that may be a list or a dict wrapping a list ─────────
def load_pages(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # unwrap filename→pages dict
        for v in data.values():
            if isinstance(v, list):
                return v
        print(f"ERROR: no list found in {path}", file=sys.stderr)
        sys.exit(1)
    return data

ocr_pages    = load_pages(OCR_JSON)
layout_pages = load_pages(LAYOUT_JSON)

# sanity check: they should have the same number of pages
if len(ocr_pages) != len(layout_pages):
    print("WARNING: OCR pages and layout pages count differ", file=sys.stderr)

# ─── 4) Write out the CSV ─────────────────────────────────────────────────────
os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow(["pg_num", "layout", "text", "confidence"])

    # enumerate pages since JSON doesn't include a 'page' field
    for pg_num, (ocr_page, layout_page) in enumerate(zip(ocr_pages, layout_pages), start=1):
        lines = ocr_page.get("text_lines", [])
        for block in layout_page.get("bboxes", []):
            bbox   = block.get("bbox", [])
            label  = block.get("label", "")
            # collect all lines whose center falls inside this block
            member_lines = [L for L in lines if line_in_block(L["bbox"], bbox)]
            if not member_lines:
                continue

            # join into a paragraph and average confidences
            texts = [L["text"].replace("\n", " ").strip() for L in member_lines]
            confs = [L.get("confidence", 0) for L in member_lines]
            paragraph = " ".join(texts)
            confidence = sum(confs) / len(confs) if confs else ""

            writer.writerow([pg_num, label, paragraph, confidence])

print(f"Wrote CSV to {CSV_OUTPUT}")
