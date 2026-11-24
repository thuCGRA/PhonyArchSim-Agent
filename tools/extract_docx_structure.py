from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from docx import Document
from docx.document import Document as _Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_blocks(parent):
    if not isinstance(parent, _Document):
        raise TypeError("Only top-level document bodies are supported")
    for child in parent.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def paragraph_record(index: int, paragraph: Paragraph) -> dict:
    runs = []
    for run in paragraph.runs:
        runs.append(
            {
                "text": run.text,
                "bold": bool(run.bold),
                "italic": bool(run.italic),
                "underline": bool(run.underline),
            }
        )
    return {
        "kind": "paragraph",
        "index": index,
        "style": paragraph.style.name if paragraph.style else None,
        "text": paragraph.text,
        "runs": runs,
    }


def table_record(index: int, table: Table) -> dict:
    return {
        "kind": "table",
        "index": index,
        "rows": [
            [cell.text.replace("\n", " / ") for cell in row.cells]
            for row in table.rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    document = Document(args.input)
    blocks = []
    paragraph_index = 0
    table_index = 0
    for block in iter_blocks(document):
        if isinstance(block, Paragraph):
            blocks.append(paragraph_record(paragraph_index, block))
            paragraph_index += 1
        else:
            blocks.append(table_record(table_index, block))
            table_index += 1

    with zipfile.ZipFile(args.input) as archive:
        document_xml = archive.read("word/document.xml")
        media = sorted(
            name for name in archive.namelist() if name.startswith("word/media/")
        )

    payload = {
        "source": str(args.input),
        "paragraph_count": paragraph_index,
        "table_count": table_index,
        "inline_shape_count": len(document.inline_shapes),
        "omath_count": document_xml.count(b"<m:oMath"),
        "drawing_count": document_xml.count(b"<w:drawing"),
        "pict_count": document_xml.count(b"<w:pict"),
        "media": media,
        "blocks": blocks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
