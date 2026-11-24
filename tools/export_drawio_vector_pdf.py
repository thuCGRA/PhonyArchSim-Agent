from __future__ import annotations

import argparse
import base64
import subprocess
import tempfile
import time
import urllib.parse
import zlib
from pathlib import Path

from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter


def drawio_fragment(xml: str) -> str:
    encoded = urllib.parse.quote(xml, safe="~()*!.'")
    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(encoded.encode("utf-8")) + compressor.flush()
    payload = base64.b64encode(compressed).decode("ascii")
    return urllib.parse.quote(payload, safe="")


def content_bbox(image_path: Path) -> tuple[int, int, int, int]:
    image = Image.open(image_path).convert("RGB")
    background = Image.new("RGB", image.size, "white")
    difference = ImageChops.difference(image, background).convert("L")
    difference = difference.point(lambda value: 255 if value > 8 else 0)
    bbox = difference.getbbox()
    if bbox is None:
        raise RuntimeError("The browser-generated PDF rendered as a blank page")
    return bbox


def crop_pdf_to_rendered_content(
    raw_pdf: Path, rendered_png: Path, output_pdf: Path, padding_pt: float = 5.0
) -> None:
    reader = PdfReader(raw_pdf)
    if len(reader.pages) != 1:
        raise RuntimeError(f"Expected one printed page, found {len(reader.pages)}")
    page = reader.pages[0]
    image = Image.open(rendered_png)
    left_px, top_px, right_px, bottom_px = content_bbox(rendered_png)

    page_width = float(page.mediabox.width)
    page_height = float(page.mediabox.height)
    scale_x = page_width / image.width
    scale_y = page_height / image.height

    left = max(float(page.mediabox.left), left_px * scale_x - padding_pt)
    right = min(float(page.mediabox.right), right_px * scale_x + padding_pt)
    bottom = max(
        float(page.mediabox.bottom), page_height - bottom_px * scale_y - padding_pt
    )
    top = min(float(page.mediabox.top), page_height - top_px * scale_y + padding_pt)

    page.mediabox.lower_left = (left, bottom)
    page.mediabox.upper_right = (right, top)
    page.cropbox.lower_left = (left, bottom)
    page.cropbox.upper_right = (right, top)

    writer = PdfWriter()
    writer.add_page(page)
    with output_pdf.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--edge", type=Path, required=True)
    parser.add_argument("--pdftoppm", default="pdftoppm")
    args = parser.parse_args()

    xml = args.input.read_text(encoding="utf-8")
    url = (
        "https://viewer.diagrams.net/?highlight=0000ff&layers=1&nav=0&title=figure"
        f"#R{drawio_fragment(xml)}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="codex_drawio_pdf_") as temp_dir:
        temp = Path(temp_dir)
        raw_pdf = temp / "raw.pdf"
        render_prefix = temp / "render"
        command = [
            str(args.edge),
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--no-first-run",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=8000",
            f"--user-data-dir={temp / 'edge-profile'}",
            f"--print-to-pdf={raw_pdf}",
            url,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stable_polls = 0
        previous_size = -1
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if raw_pdf.exists() and raw_pdf.stat().st_size > 0:
                current_size = raw_pdf.stat().st_size
                if current_size == previous_size:
                    stable_polls += 1
                else:
                    stable_polls = 0
                previous_size = current_size
                if stable_polls >= 2:
                    break
            if process.poll() is not None and not raw_pdf.exists():
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"Edge PDF export failed with code {process.returncode}: "
                    f"{stdout}\n{stderr}"
                )
            time.sleep(1)
        if not raw_pdf.exists() or raw_pdf.stat().st_size == 0:
            process.terminate()
            raise RuntimeError("Edge did not produce the printed PDF within 60 seconds")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

        subprocess.run(
            [
                args.pdftoppm,
                "-f",
                "1",
                "-l",
                "1",
                "-png",
                "-r",
                "180",
                str(raw_pdf),
                str(render_prefix),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        rendered = temp / "render-1.png"
        crop_pdf_to_rendered_content(raw_pdf, rendered, args.output)


if __name__ == "__main__":
    main()
