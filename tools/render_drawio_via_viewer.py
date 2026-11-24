from __future__ import annotations

import argparse
import base64
import subprocess
import tempfile
import urllib.parse
import zlib
from pathlib import Path

from PIL import Image, ImageChops


def drawio_fragment(xml: str) -> str:
    encoded = urllib.parse.quote(xml, safe="~()*!.'")
    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(encoded.encode("utf-8")) + compressor.flush()
    payload = base64.b64encode(compressed).decode("ascii")
    return urllib.parse.quote(payload, safe="")


def crop_uniform_border(input_path: Path, output_path: Path, padding: int = 18) -> None:
    image = Image.open(input_path).convert("RGB")
    background = Image.new("RGB", image.size, image.getpixel((0, 0)))
    bbox = ImageChops.difference(image, background).getbbox()
    if bbox is None:
        raise RuntimeError("The rendered viewer screenshot is blank")
    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(image.width, bbox[2] + padding)
    bottom = min(image.height, bbox[3] + padding)
    image.crop((left, top, right, bottom)).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--edge", type=Path, required=True)
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=1800)
    args = parser.parse_args()

    xml = args.input.read_text(encoding="utf-8")
    fragment = drawio_fragment(xml)
    url = (
        "https://viewer.diagrams.net/?highlight=0000ff&layers=1&nav=0&title=figure"
        f"#R{fragment}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="codex_drawio_") as profile_dir:
        raw_path = args.output.with_name(args.output.stem + "-raw.png")
        command = [
            str(args.edge),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--disable-extensions",
            "--no-first-run",
            f"--user-data-dir={profile_dir}",
            f"--window-size={args.width},{args.height}",
            "--virtual-time-budget=8000",
            f"--screenshot={raw_path}",
            url,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if completed.returncode != 0 or not raw_path.exists():
            raise RuntimeError(
                f"Edge render failed with code {completed.returncode}: {completed.stderr}"
            )
        crop_uniform_border(raw_path, args.output)
        raw_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
