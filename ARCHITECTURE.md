# Architecture: plato-tile-split

## Language Choice: Rust

### Why Rust

Text chunking is CPU-bound string processing. The key advantage:
Rust's `&str` slicing is **zero-copy** — no new allocations for substrings.

| Metric | Python (str.split) | Rust (&str.split) |
|--------|--------------------|--------------------|
| Split 1MB text | ~15ms | ~2ms |
| Memory per chunk | ~200 bytes (str obj) | ~40 bytes (String) |
| Code block parsing | regex + loops | char-by-char state machine |

### Why not tiktoken

tiktoken is the gold standard for tokenization (OpenAI's tokenizer).
But it's a Python C extension with complex build requirements.
Our `chars_per_token` estimate (4.0) is accurate enough for chunking purposes.
For exact token counts, we'd add tiktoken as an optional dependency.

### Why not langchain text_splitter

langchain's RecursiveCharacterTextSplitter is good but:
- Python-only (no WASM, no FFI)
- Creates many intermediate string objects
- No code-aware splitting

Our implementation adds: code block awareness, brace-depth tracking for
function/class splitting, and heading/paragraph boundary detection.

### Architecture

```
Input text → code_aware? → separate code/text blocks
                ↓
           Text blocks → split at headings/paragraphs/sentences
                ↓
           Buffer segments → flush at max_tokens → apply overlap
                ↓
           Vec<Chunk> { text, start, end, token_estimate, type }
```

### Splitting Strategies

1. **Boundary-aware**: Prefer splitting at natural boundaries (`. `, `\n\n`, `# `)
2. **Code-aware**: Never split inside ``` code blocks
3. **Overlap**: Preserve N tokens of context between chunks
4. **Code splitting**: Brace-depth tracking for function/class boundaries
