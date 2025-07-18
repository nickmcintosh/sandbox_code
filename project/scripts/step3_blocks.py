# #!/usr/bin/env python3
# from pathlib import Path
# import csv, json, yaml, sys

# # ─── Resolve project root & load config ─────────────────────────────────────
# SCRIPT_ROOT = Path(__file__).resolve().parents[1]
# CONFIG_PATH = SCRIPT_ROOT / "config.yaml"
# cfg         = yaml.safe_load(CONFIG_PATH.read_text())

# # ─── Directories ────────────────────────────────────────────────────────────
# OCR_DIR     = SCRIPT_ROOT / cfg["ocr_dir"]
# BLOCKS_DIR  = SCRIPT_ROOT / cfg["blocks_dir"]

# # helper: test whether the center of a text-line bbox falls in a block bbox
# def line_in_block(line_bbox, block_bbox):
#     lx1, ly1, lx2, ly2 = line_bbox
#     bx1, by1, bx2, by2 = block_bbox
#     cx, cy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
#     return bx1 <= cx <= bx2 and by1 <= cy <= by2


# def json_to_csv(doc_name=None):
#     """
#     Read OCR+layout JSON for a document and write blocks.csv with columns:
#     book_title, pg_num, layout, text, confidence
#     """
#     BLOCKS_DIR.mkdir(parents=True, exist_ok=True)

#     # If a specific document is given, limit to that; else process all
#     doc_dirs = [Path(document) for document in ([doc_name] if doc_name else [])] if doc_name else list(sorted(OCR_DIR.iterdir()))
#     for doc_dir in list(sorted(OCR_DIR.iterdir())):
#         if not doc_dir.is_dir():
#             continue
#         name = doc_dir.name
#         if doc_name and name != doc_name:
#             continue

#         ocr_path    = doc_dir / "ocr_results.jsonl"
#         layout_path = doc_dir / "layout_results.jsonl"
#         out_dir     = BLOCKS_DIR / name
#         out_dir.mkdir(parents=True, exist_ok=True)
#         csv_path    = out_dir / "blocks.csv"

#         # load JSON data
#         with open(ocr_path, encoding="utf-8") as f:
#             ocr_pages = json.load(f)
#         with open(layout_path, encoding="utf-8") as f:
#             layout_pages = json.load(f)

#         # write CSV with new book_title column first
#         with open(csv_path, "w", newline="", encoding="utf-8") as cf:
#             writer = csv.writer(cf)
#             writer.writerow(["book_title", "pg_num", "layout", "text", "confidence"])

#             for pg_num, (ocr_page, layout_page) in enumerate(zip(ocr_pages, layout_pages), start=1):
#                 lines = ocr_page.get("text_lines", [])
#                 for block in layout_page.get("bboxes", []):
#                     bbox       = block.get("bbox", [])
#                     label      = block.get("label", "")
#                     members    = [L for L in lines if line_in_block(L.get("bbox", []), bbox)]
#                     if not members:
#                         continue

#                     # assemble paragraph text and average confidence
#                     texts       = [L.get("text", "").replace("\n", " ").strip() for L in members]
#                     confs       = [L.get("confidence", 0) for L in members]
#                     paragraph   = " ".join(texts)
#                     confidence  = sum(confs) / len(confs) if confs else ""

#                     # write row with book_title first
#                     writer.writerow([name, pg_num, label, paragraph, confidence])

#         print(f"✅ Wrote blocks.csv for '{name}' at {csv_path}")
#     print("[step3] All documents processed.")


from pathlib import Path
import os, csv, json, yaml

# ─── Config & Paths ─────────────────────────────────────────────────────────
SCRIPT_ROOT = Path(__file__).resolve().parent.parent
cfg         = yaml.safe_load((SCRIPT_ROOT / "config.yaml").read_text())

OCR_DIR     = SCRIPT_ROOT / cfg["ocr_dir"]
BLOCKS_DIR  = SCRIPT_ROOT / cfg["blocks_dir"]

def line_in_block(line_bbox, block_bbox):
    lx1, ly1, lx2, ly2 = line_bbox
    bx1, by1, bx2, by2 = block_bbox
    cx, cy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
    return bx1 <= cx <= bx2 and by1 <= cy <= by2

def jsonl_to_list(path: Path):
    """Read a .jsonl file and return a list of parsed JSON objects."""
    pages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pages.append(json.loads(line))
    return pages

def json_to_csv(doc_name: str = None):
    """
    Read OCR+layout NDJSON and write blocks.csv with columns:
    book_title, pg_num, layout, text, confidence
    """
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which documents to process
    if doc_name:
        doc_dirs = [OCR_DIR / doc_name]
    else:
        doc_dirs = sorted([d for d in OCR_DIR.iterdir() if d.is_dir()])

    for doc_dir in doc_dirs:
        name = doc_dir.name

        ocr_path    = doc_dir / "ocr_results.jsonl"
        layout_path = doc_dir / "layout_results.jsonl"
        out_dir     = BLOCKS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path    = out_dir / "blocks.csv"

        if not ocr_path.exists() or not layout_path.exists():
            print(f"[step3] ⚠️  Skipping '{name}': missing JSONL files")
            continue

        # Load pages from NDJSON
        ocr_pages    = jsonl_to_list(ocr_path)
        layout_pages = jsonl_to_list(layout_path)

        # Write combined CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["book_title", "pg_num", "layout", "text", "confidence"])

            for pg_num, (ocr_page, layout_page) in enumerate(zip(ocr_pages, layout_pages), start=1):
                lines = ocr_page.get("text_lines", [])
                for block in layout_page.get("bboxes", []):
                    bbox  = block.get("bbox", [])
                    label = block.get("label", "")
                    # find all text_lines whose center falls inside this block
                    members = [L for L in lines if line_in_block(L["bbox"], bbox)]
                    if not members:
                        continue
                    # join into one paragraph
                    texts = [L["text"].replace("\n", " ").strip() for L in members]
                    confs = [L.get("confidence", 0) for L in members]
                    paragraph = " ".join(texts)
                    confidence = sum(confs) / len(confs) if confs else ""

                    writer.writerow([name, pg_num, label, paragraph, confidence])

        print(f"[step3] ✅ Wrote blocks.csv for '{name}'")

# If you need this to be callable via CLI:
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate OCR+layout JSONL into CSV blocks.")
    ap.add_argument("--doc_name", help="Only process this subfolder")
    args = ap.parse_args()
    json_to_csv(args.doc_name)

