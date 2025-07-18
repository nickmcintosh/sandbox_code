# scripts/init_dirs.py
from pathlib import Path
import yaml

def init_dirs(config_path="config.yaml"):
    """
    Read config.yaml and mkdir -p every data directory plus scripts/.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    # base dirs from config
    raw       = Path(cfg["input_dir"]) / "raw"
    images    = Path(cfg["image_dir"])
    ocr       = Path(cfg["ocr_dir"])
    blocks    = Path(cfg["blocks_dir"])
    cleaned   = Path(cfg["cleaned_dir"])
    # optional: ensure scripts/ exists too
    scripts   = Path("scripts")

    for d in (raw, images, ocr, blocks, cleaned, scripts):
        d.mkdir(parents=True, exist_ok=True)
