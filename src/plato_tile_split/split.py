"""Tile decomposition."""
import re
from dataclasses import dataclass, field

@dataclass
class SubTile:
    content: str
    parent_id: str
    index: int
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)

class TileSplitter:
    def __init__(self, min_chunk_size: int = 50, max_chunk_size: int = 500):
        self.min_chunk = min_chunk_size
        self.max_chunk = max_chunk_size

    def by_sentence(self, content: str, parent_id: str = "") -> list[SubTile]:
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        chunks, current = [], []
        for s in sentences:
            current.append(s)
            if len(" ".join(current)) >= self.min_chunk:
                chunks.append(" ".join(current))
                current = []
        if current: chunks.append(" ".join(current))
        return [SubTile(content=c, parent_id=parent_id, index=i) for i, c in enumerate(chunks) if c.strip()]

    def by_paragraph(self, content: str, parent_id: str = "") -> list[SubTile]:
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        return [SubTile(content=p, parent_id=parent_id, index=i) for i, p in enumerate(paragraphs) if len(p) >= self.min_chunk]

    def by_size(self, content: str, parent_id: str = "", size: int = 0) -> list[SubTile]:
        size = size or self.max_chunk
        chunks = [content[i:i+size] for i in range(0, len(content), size)]
        return [SubTile(content=c, parent_id=parent_id, index=i) for i, c in enumerate(chunks) if c.strip()]

    def auto(self, content: str, parent_id: str = "") -> list[SubTile]:
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if all(len(p) <= self.max_chunk for p in paragraphs) and len(paragraphs) > 1:
            return self.by_paragraph(content, parent_id)
        if len(content) <= self.max_chunk:
            return [SubTile(content=content, parent_id=parent_id, index=0)]
        return self.by_size(content, parent_id)
