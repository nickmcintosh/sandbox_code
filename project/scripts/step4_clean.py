### scripts/step4_clean.py
#!/usr/bin/env python3
from pathlib import Path
import re, csv, yaml, sys

# ─── Resolve project root & load config ─────────────────────────────────────
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = SCRIPT_ROOT / "config.yaml"
cfg         = yaml.safe_load(CONFIG_PATH.read_text())

# ─── Directories ────────────────────────────────────────────────────────────
BLOCKS_DIR  = SCRIPT_ROOT / cfg["blocks_dir"]
CLEAN_DIR   = SCRIPT_ROOT / cfg["cleaned_dir"]

# regex rules
HYPHEN_SPACE_RULE = re.compile(r"-\s+")

# replacement mapping
def clean_text(text: str) -> str:
    # 1) remove hyphen + space
    text = HYPHEN_SPACE_RULE.sub("", text)
    # 2) replace guillemets
    text = text.replace("<<", "«").replace(">>", "»")
    return text

def run_clean():
    print(f"[step4] BLOCKS_DIR = {BLOCKS_DIR!r}, exists? {BLOCKS_DIR.exists()}")
    docs = [d for d in sorted(BLOCKS_DIR.iterdir()) if d.is_dir()]
    print(f"[step4] found {len(docs)} doc dir(s): {[d.name for d in docs]}")
    if not docs:
        print("[step4] 🚨 No block CSVs to clean.", file=sys.stderr)
        return

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    for doc_dir in docs:
        name     = doc_dir.name
        in_csv   = doc_dir / "blocks.csv"
        out_dir  = CLEAN_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv  = out_dir / "blocks_clean.csv"

        if not in_csv.exists():
            print(f"[step4]    ✖ Missing blocks.csv for '{name}'", file=sys.stderr)
            continue

        print(f"[step4] → Cleaning '{name}', writing to {out_csv}")
        with (
            open(in_csv, newline="", encoding="utf-8") as rf,
            open(out_csv, "w", newline="", encoding="utf-8") as wf
        ):
            reader = csv.DictReader(rf)
            writer = csv.DictWriter(wf, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                row["text"] = clean_text(row.get("text", ""))
                writer.writerow(row)
