#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR pipeline using PaddleOCR PPStructureV3.

Processes either:
  • A PDF at ocr/input/PDF/KAFKA_FR-5-9.pdf  (merged into one .md + JSON per page)
  • Or a directory of JPEGs under ocr/input/JPG/KAFKA_FR-5-9/*.jpg

Outputs go to ocr/output/KAFKA_FR-5-9/.
"""
import sys
from pathlib import Path
from paddleocr import PPStructureV3

def ocr_pdf(pdf_path: Path, output_dir: Path):
    pipeline = PPStructureV3()  # instantiate the structure pipeline :contentReference[oaicite:0]{index=0}
    results = pipeline.predict(input=str(pdf_path))
    # save per-page JSON/Markdown
    for idx, res in enumerate(results, start=1):
        res.save_to_json(save_path=str(output_dir))
        res.save_to_markdown(save_path=str(output_dir))
    # combine all pages into one Markdown document
    md_pages = [res.markdown for res in results]
    combined_md = pipeline.concatenate_markdown_pages(md_pages)
    md_file = output_dir / f"{pdf_path.stem}.md"
    md_file.write_text(combined_md, encoding="utf-8")

def ocr_images(img_dir: Path, output_dir: Path):
    pipeline = PPStructureV3()
    # process each JPG in sorted order
    for img in sorted(img_dir.glob("*.jpg")):
        results = pipeline.predict(input=str(img))
        for res in results:
            res.save_to_json(save_path=str(output_dir))
            res.save_to_markdown(save_path=str(output_dir))

def main():
    pdf_path = Path("ocr/input/PDF/KAFKA_FR-5-9.pdf")
    img_dir   = Path("ocr/input/JPG/KAFKA_FR-5-9")
    output_dir = Path("ocr/output/KAFKA_FR-5-9")
    output_dir.mkdir(parents=True, exist_ok=True)

    if pdf_path.is_file():
        print(f"→ OCR’ing PDF: {pdf_path}")
        ocr_pdf(pdf_path, output_dir)
    elif img_dir.is_dir():
        print(f"→ OCR’ing images in: {img_dir}")
        ocr_images(img_dir, output_dir)
    else:
        print(f"Error: Neither PDF ({pdf_path}) nor JPG folder ({img_dir}) found.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
