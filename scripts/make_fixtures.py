"""Generate the fixture contract PDFs from markdown-ish source files.

Each source file in ``data/contracts/src/*.md`` is split into pages on a line
containing only ``===PAGE===``. Every page is rendered with reportlab so the
text layer is clean and the page breaks are deterministic (important for the
grounding and page-accuracy checks).

Contracts listed in ``SCANNED`` are rendered as page *images* with no text
layer, like a scanned paper copy: the intake step must detect that and send the
contract to manual review without calling the model.

Usage (from the repo root, inside the venv):
    python scripts/make_fixtures.py                 # all contracts
    python scripts/make_fixtures.py contract_f contract_g
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "contracts" / "src"
OUT = ROOT / "data" / "contracts"

PAGE_BREAK = "===PAGE==="
FONT = "Helvetica"
SIZE = 10.5
LEADING = 15
LEFT = 2.2 * cm
TOP = 28.0 * cm
BOTTOM = 2.5 * cm
MAX_WIDTH = A4[0] - 2 * LEFT
SCANNED = {"contract_g"}


def _wrap(c: canvas.Canvas, text: str, font: str, size: float) -> list[str]:
    """Greedy word-wrap a single logical line to the page width."""
    if not text:
        return [""]
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if c.stringWidth(trial, font, size) <= MAX_WIDTH:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _render_page(c: canvas.Canvas, body: str) -> None:
    y = TOP
    for raw_line in body.splitlines():
        bold = raw_line.startswith("# ") or raw_line.startswith("## ")
        line = raw_line[2:] if bold else raw_line
        font = "Helvetica-Bold" if bold else FONT
        for seg in _wrap(c, line, font, SIZE):
            if y < BOTTOM:
                break
            c.setFont(font, SIZE)
            c.drawString(LEFT, y, seg)
            y -= LEADING
    c.showPage()


def build(src_path: Path) -> Path:
    text = src_path.read_text(encoding="utf-8")
    pages = [p.strip("\n") for p in text.split(PAGE_BREAK)]
    out_path = OUT / (src_path.stem + ".pdf")
    c = canvas.Canvas(str(out_path), pagesize=A4)
    for body in pages:
        _render_page(c, body)
    c.save()
    return out_path


def build_scanned(src_path: Path) -> Path:
    """Image-only PDF (no text layer), 150 dpi A4 pages with slight grey 'paper'."""
    import textwrap

    from PIL import Image, ImageDraw, ImageFont

    w, h = 1240, 1754
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        try:
            font = ImageFont.load_default(size=24)
        except TypeError:
            font = ImageFont.load_default()
    images = []
    for body in src_path.read_text(encoding="utf-8").split(PAGE_BREAK):
        img = Image.new("L", (w, h), 242)
        d = ImageDraw.Draw(img)
        y = 140
        for raw in body.strip("\n").splitlines():
            line = raw.lstrip("# ")
            for seg in textwrap.wrap(line, 78) or [""]:
                d.text((130, y), seg, fill=40, font=font)
                y += 36
        images.append(img.rotate(0.6, fillcolor=242).convert("RGB"))
    out_path = OUT / (src_path.stem + ".pdf")
    images[0].save(out_path, "PDF", resolution=150, save_all=True, append_images=images[1:])
    return out_path


def main() -> None:
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    srcs = [p for p in sorted(SRC.glob("*.md")) if not only or p.stem in only]
    if not srcs:
        print(f"No source files in {SRC}")
        return
    for src in srcs:
        out = build_scanned(src) if src.stem in SCANNED else build(src)
        print(f"wrote {out.relative_to(ROOT)}  ({len(out.read_bytes())} bytes)")


if __name__ == "__main__":
    main()
