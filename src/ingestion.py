from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import re

import fitz


@dataclass(frozen=True)
class PageDocument:
    text: str
    source_file: str
    volume: str
    page: int

    def to_dict(self) -> dict:
        return asdict(self)


def _volume_from_filename(path: Path) -> str:
    match = re.search(r"Volume[_\s-]*(\d+)", path.stem, flags=re.IGNORECASE)
    return f"Volume {match.group(1)}" if match else path.stem


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(data_dir: str | Path = "Data") -> list[PageDocument]:
    data_path = Path(data_dir)
    pdf_paths = sorted(data_path.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_path.resolve()}")

    pages: list[PageDocument] = []
    for pdf_path in pdf_paths:
        volume = _volume_from_filename(pdf_path)
        with fitz.open(pdf_path) as doc:
            for page_index in range(doc.page_count):
                text = clean_text(doc[page_index].get_text("text"))
                if text:
                    pages.append(
                        PageDocument(
                            text=text,
                            source_file=pdf_path.name,
                            volume=volume,
                            page=page_index + 1,
                        )
                    )
    return pages
