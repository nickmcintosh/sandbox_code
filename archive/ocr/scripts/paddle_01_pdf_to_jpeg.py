#!/usr/bin/env python3
"""
Convert all pages of a PDF into JPEGs at 300 DPI.
Edit the two constants below, then hit Run in VS Code.
"""

import os
import fitz  # PyMuPDF
from pathlib import Path

# ── EDIT THESE ────────────────────────────────────────────────────────────────
FILE_NAME = "KAFKA_FR-5-9"
PDF_PATH   = Path(f"ocr/input/PDF/{FILE_NAME}.pdf")   # your source PDF
OUTPUT_DIR = Path(f"ocr/input/JPG/{FILE_NAME}.jpg")              # folder to save JPEGs
DPI        = 500                                   # image resolution
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_jpg(pdf_path: Path, out_dir: Path, dpi: int = 300):
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        mat = fitz.Matrix(dpi/72, dpi/72)  # scale from 72→dpi
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"page_{page_num:03d}.jpg"
        pix.save(out_path)
        print(f"Saved {out_path}")
    doc.close()

if __name__ == "__main__":
    pdf_to_jpg(PDF_PATH, OUTPUT_DIR, DPI)