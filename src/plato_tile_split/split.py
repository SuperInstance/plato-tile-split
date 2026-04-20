"""Tile decomposition — sentence, paragraph, word, markdown, and auto splitting."""
import re
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SubTile:
    content: str
    parent_id: str
    index: int
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

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
        if current:
            chunks.append(" ".join(current))
        return [SubTile(content=c, parent_id=parent_id, index=i,
                        tags=["sentence-split"]) for i, c in enumerate(chunks) if c.strip()]

    def by_paragraph(self, content: str, parent_id: str = "") -> list[SubTile]:
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        return [SubTile(content=p, parent_id=parent_id, index=i,
                        tags=["paragraph-split"]) for i, p in enumerate(paragraphs)
                if len(p) >= self.min_chunk or i == 0]

    def by_word(self, content: str, parent_id: str = "", words: int = 0) -> list[SubTile]:
        words = words or max(10, self.max_chunk // 5)
        word_list = content.split()
        chunks = []
        for i in range(0, len(word_list), words):
            chunk = " ".join(word_list[i:i + words])
            if chunk.strip():
                chunks.append(chunk)
        return [SubTile(content=c, parent_id=parent_id, index=i,
                        tags=["word-split"]) for i, c in enumerate(chunks)]

    def by_size(self, content: str, parent_id: str = "", size: int = 0) -> list[SubTile]:
        size = size or self.max_chunk
        chunks = [content[i:i + size] for i in range(0, len(content), size)]
        return [SubTile(content=c, parent_id=parent_id, index=i,
                        tags=["size-split"]) for i, c in enumerate(chunks) if c.strip()]

    def by_markdown(self, content: str, parent_id: str = "") -> list[SubTile]:
        """Split on markdown headers (##, ###, etc.)"""
        sections = re.split(r'\n(?=#{1,3}\s)', content.strip())
        return [SubTile(content=s.strip(), parent_id=parent_id, index=i,
                        tags=["markdown-split"])
                for i, s in enumerate(sections) if s.strip()]

    def by_code_block(self, content: str, parent_id: str = "") -> list[SubTile]:
        """Extract code blocks and prose separately."""
        tiles = []
        blocks = re.split(r'(```\w*\n.*?```)', content, flags=re.DOTALL)
        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue
            tag = "code-block" if block.startswith("```") else "prose"
            tiles.append(SubTile(content=block, parent_id=parent_id, index=i, tags=[tag]))
        return tiles

    def auto(self, content: str, parent_id: str = "") -> list[SubTile]:
        """Auto-detect best splitting strategy."""
        has_headers = bool(re.search(r'^#{1,3}\s', content, re.MULTILINE))
        has_code = "```" in content
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        if has_code and len(content) > self.max_chunk:
            return self.by_code_block(content, parent_id)
        if has_headers:
            return self.by_markdown(content, parent_id)
        if all(len(p) <= self.max_chunk for p in paragraphs) and len(paragraphs) > 1:
            return self.by_paragraph(content, parent_id)
        if len(content) <= self.max_chunk:
            return [SubTile(content=content, parent_id=parent_id, index=0, tags=["atomic"])]
        return self.by_size(content, parent_id)

    def estimate_chunks(self, content: str, strategy: str = "auto") -> int:
        if strategy == "sentence":
            return max(1, len(re.findall(r'[.!?]', content)) // max(1, self.min_chunk // 20))
        elif strategy == "paragraph":
            return max(1, len(content.split('\n\n')))
        elif strategy == "word":
            return max(1, len(content.split()) // max(1, self.max_chunk // 5))
        return max(1, len(content) // self.max_chunk)

    @property
    def stats(self) -> dict:
        return {"min_chunk": self.min_chunk, "max_chunk": self.max_chunk,
                "strategies": ["sentence", "paragraph", "word", "size", "markdown", "code_block", "auto"]}
