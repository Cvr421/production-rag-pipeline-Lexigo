import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_file: str
    chunk_index: int
    text: str


def _normalise_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()


def _split_long_section(section: str, chunk_size: int, overlap: int) -> list[str]:
    if len(section) <= chunk_size:
        return [section]

    paragraphs = [part.strip() for part in section.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)

    if not chunks:
        start = 0
        while start < len(section):
            end = min(start + chunk_size, len(section))
            chunks.append(section[start:end].strip())
            if end == len(section):
                break
            start = max(end - overlap, start + 1)
    return chunks


def split_markdown(
    text: str,
    source_file: str,
    chunk_size: int = 1200,
    overlap: int = 160,
) -> list[DocumentChunk]:
    """Split Markdown by headings first, then by paragraphs for long sections."""
    normalised = _normalise_text(text)
    sections = [
        part.strip()
        for part in re.split(r"(?=^#{1,6}\s+)", normalised, flags=re.MULTILINE)
    ]
    sections = [section for section in sections if section]
    pieces: list[str] = []
    for section in sections:
        pieces.extend(_split_long_section(section, chunk_size, overlap))

    chunks: list[DocumentChunk] = []
    for index, piece in enumerate(pieces):
        digest = hashlib.sha1(f"{source_file}\n{index}\n{piece}".encode("utf-8")).hexdigest()[:12]
        stem = Path(source_file).stem
        chunks.append(
            DocumentChunk(
                chunk_id=f"{stem}-{index:03d}-{digest}",
                source_file=source_file,
                chunk_index=index,
                text=piece,
            )
        )
    return chunks


def load_markdown_documents(corpus_dir: str | Path) -> list[DocumentChunk]:
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")

    chunks: list[DocumentChunk] = []
    for path in sorted(root.rglob("*.md")):
        relative_path = path.relative_to(root).as_posix()
        chunks.extend(split_markdown(path.read_text(encoding="utf-8"), relative_path))
    if not chunks:
        raise ValueError(f"No Markdown documents found in {root}")
    return chunks
