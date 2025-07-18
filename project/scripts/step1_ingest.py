# scripts/step1_ingest.py
from pathlib import Path
from pdf2image import convert_from_path
import yaml

# load config once
SCRIPT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = SCRIPT_ROOT / "config.yaml"

cfg = yaml.safe_load(CONFIG_PATH.read_text())
RAW_DIR    = Path(cfg["input_dir"]) / "raw"
IMAGES_DIR = Path(cfg["image_dir"])

def ingest_pdfs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for pdf in RAW_DIR.glob("*.pdf"):
        docname = pdf.stem
        print(f"Initializing {docname}.")
        outdir  = IMAGES_DIR / docname
        outdir.mkdir(exist_ok=True)
        pages = convert_from_path(
            str(pdf),
            dpi=cfg["dpi"],
            thread_count=cfg["pdf2image_threads"]
        )
        for i, img in enumerate(pages, start=1):
            img.save(outdir / f"page_{i:03d}.png")
            print(f"{docname} - pg. {i} saved")

