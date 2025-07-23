#!/usr/bin/env python3
"""
OCR pipeline commands.
"""
import click
import yaml
from pathlib import Path
from scripts.init_dirs import init_dirs
from scripts.step1_ingest import ingest_pdfs
from scripts.step2_ocr import run_batch, get_device
from scripts.step3_blocks import json_to_csv
from scripts.step4_clean import run_clean

@click.group()
def cli():
    """OCR pipeline commands."""
    pass

@cli.command()
def init():
    """Step 0: create needed dirs"""
    init_dirs()
    click.echo("✅ Directories created")

@cli.command()
def images():
    """Step 1: PDF → PNG"""
    ingest_pdfs()
    click.echo("✅ Images generated")

@cli.command()
@click.option("--doc_name", default=None, help="Only this subfolder")
@click.option("--cpu_only", is_flag=True, help="CPU only mode")
@click.option("--downscale", default=1.0, type=float, help="Resize factor (e.g. 0.5)")
@click.option("--batch_size", default=0, type=int, help="Pages per batch; 0 disables batching")
def ocr(doc_name, cpu_only, downscale, batch_size):
    """Step 2: Run OCR in page batches"""
    # load config (not used for images_dir override)
    yaml.safe_load(Path("config.yaml").read_text())
    # point to project/data/images
    images_dir = Path("data") / "images"
    docs = [doc_name] if doc_name else sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    for name in docs:
        total = len(list((images_dir / name).glob("*.png")))
        if batch_size > 0:
            starts = list(range(1, total + 1, batch_size))
        else:
            starts = [1]
            batch_size = total
        for start in starts:
            end = min(start + batch_size - 1, total)
            click.echo(f"[pipeline] Running batch {start}-{end} for '{name}'")
            try:
                device = get_device(cpu_only)
                run_batch(
                    name=name,
                    device=device,
                    downscale=downscale,
                    start_page=start,
                    end_page=end
                )
            except Exception as e:
                click.echo(f"⚠️  Batch {start}-{end} for '{name}' failed: {e}. Skipping.")
                continue
    click.echo("✅ OCR complete.")

@cli.command()
def blocks():
    """Step 3: blocks → CSV"""
    json_to_csv()
    click.echo("✅ Blocks CSV done")

@cli.command()
def clean():
    """Step 4: regex clean"""
    run_clean()
    click.echo("✅ Text cleaned")

@cli.command()
@click.pass_context
def all(ctx):
    """Run all steps"""
    init_dirs()
    ingest_pdfs()
    ctx.invoke(ocr)
    json_to_csv()
    run_clean()
    click.echo("✅ Full pipeline done")

if __name__ == "__main__":
    cli()
