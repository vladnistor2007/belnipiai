import re
from typing import Generator

CHUNK_SIZE    = 700
CHUNK_OVERLAP = 200
EXCEL_CHUNK_SIZE = 420
EXCEL_OVERLAP_ROWS = 1


def _estimate_tokens(text: str) -> int:
    cyrillic = sum(1 for c in text if 'Ѐ' <= c <= 'ӿ')
    return max(1, cyrillic // 2 + (len(text) - cyrillic) // 3)


# Matches the start of a numbered or bulleted list item
_LIST_ITEM_RE = re.compile(r'^(\d+[\.\)]\s+|[•·▪▸]\s+|[–—\-]\s+(?=\S))')


def _split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into semantic paragraph units.

    Rules:
    - 2+ consecutive newlines  → definite paragraph boundary
    - Single newline before a list item (1. / • / –) → soft boundary
    - Single newlines within flowing text (PDF word-wrap) → joined with space
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    raw_blocks = re.split(r'\n{2,}', text)

    paragraphs: list[str] = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        lines = [ln.strip() for ln in block.split('\n') if ln.strip()]
        if not lines:
            continue

        current: list[str] = []
        for line in lines:
            # Start a new paragraph unit on each list item boundary
            if _LIST_ITEM_RE.match(line) and current:
                paragraphs.append(' '.join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            paragraphs.append(' '.join(current))

    return paragraphs


def _split_sentences(text: str) -> list[str]:
    """
    Split text into whole sentences — never breaks mid-sentence.
    Handles common Russian sentence endings: . ! ? ; » and ellipsis.
    """
    # Split after sentence-ending punctuation followed by whitespace + uppercase
    # or end-of-string.  We use a simple lookbehind to stay safe.
    parts = re.split(r'(?<=[.!?;»…])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _build_overlap(units: list[str], budget: int) -> list[str]:
    """
    Return the trailing units from `units` that fit within `budget` tokens.
    If no single unit fits, returns an empty list (no overlap).
    """
    tail: list[str] = []
    used = 0
    for u in reversed(units):
        t = _estimate_tokens(u)
        if used + t > budget:
            break
        tail.insert(0, u)
        used += t
    return tail


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Generator[dict, None, None]:
    """
    Paragraph-first chunking:

    1. Text is split into semantic paragraphs (double-newline boundaries,
       list item boundaries).
    2. Each paragraph is an ATOMIC unit — it is never split mid-paragraph.
    3. Only if a paragraph exceeds chunk_size on its own does it get broken
       into sentences.  Each sentence is also ATOMIC.
    4. Units are greedily accumulated into chunks.  When the next unit would
       overflow, the current chunk is flushed and overlap units are carried
       forward — always at paragraph/sentence boundaries, never mid-sentence.
    """
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return

    # Flatten paragraphs → atomic units
    # A paragraph that fits in chunk_size is kept whole.
    # A paragraph that is too long is broken into sentences.
    units: list[str] = []
    for para in paragraphs:
        if _estimate_tokens(para) <= chunk_size:
            units.append(para)
        else:
            for sentence in _split_sentences(para):
                units.append(sentence)
            # Note: if even a single sentence exceeds chunk_size it stays whole —
            # we cannot split a sentence without losing meaning.

    if not units:
        return

    chunk_idx     = 0
    current: list[str] = []
    current_tokens = 0
    i = 0

    while i < len(units):
        unit        = units[i]
        unit_tokens = _estimate_tokens(unit)

        if current and current_tokens + unit_tokens > chunk_size:
            # --- flush current chunk ---
            yield {
                "content":     "\n\n".join(current),
                "chunk_index": chunk_idx,
                "token_count": current_tokens,
            }
            chunk_idx += 1

            # Carry overlap forward
            tail       = _build_overlap(current, overlap)
            tail_tokens = sum(_estimate_tokens(u) for u in tail)

            # If overlap + incoming unit still overflows, drop the overlap to
            # prevent an infinite retry loop (this only happens for very large
            # single units that exceed chunk_size on their own).
            if tail and tail_tokens + unit_tokens > chunk_size:
                current, current_tokens = [], 0
            else:
                current, current_tokens = tail, tail_tokens

            # Do NOT advance i — retry this unit against the new (overlap) base.

        else:
            current.append(unit)
            current_tokens += unit_tokens
            i += 1

    if current:
        yield {
            "content":     "\n\n".join(current),
            "chunk_index": chunk_idx,
            "token_count": current_tokens,
        }


def chunk_excel(
    sheets: list[dict],
    chunk_size: int = EXCEL_CHUNK_SIZE,
    overlap_rows: int = EXCEL_OVERLAP_ROWS,
    source_name: str = "",
) -> Generator[dict, None, None]:
    """
    Chunk Excel/tabular data for RAG indexing.

    Each row is converted to "Col1: val1 | Col2: val2 | ..." so every chunk
    is self-contained. Each chunk starts with sheet name + column list so the
    model always knows the context even when headers are far away.

    sheets: list of {"name": str, "headers": list[str], "rows": list[dict|list]}
    """
    chunk_idx = 0

    for sheet in sheets:
        name    = sheet["name"]
        headers = [str(h).strip() or f"Колонка {i+1}" for i, h in enumerate(sheet["headers"])]
        rows    = sheet["rows"]

        if not headers or not rows:
            continue

        seen_headers: dict[str, int] = {}
        normalized_headers: list[str] = []
        for h in headers:
            base = re.sub(r"\s+", " ", h).strip() or "Колонка"
            seen_headers[base] = seen_headers.get(base, 0) + 1
            normalized_headers.append(base if seen_headers[base] == 1 else f"{base} {seen_headers[base]}")
        headers = normalized_headers

        def fmt_val(v) -> str:
            if v is None:
                return ""
            try:
                if isinstance(v, float) and v != v:
                    return ""
            except Exception:
                pass
            if hasattr(v, "strftime"):
                return v.strftime("%d.%m.%Y")
            s = str(v).strip()
            s = re.sub(r"\s+", " ", s)
            return "" if s in ("nan", "None", "NaT", "") else s

        def unpack_row(row) -> tuple[int | None, list]:
            if isinstance(row, dict):
                return row.get("row_number"), row.get("values", [])
            return None, row

        def row_to_text(row) -> str:
            row_num, values = unpack_row(row)
            parts = []
            for h, v in zip(headers, values):
                val = fmt_val(v)
                if val:
                    parts.append(f"{h}: {val}")
            if not parts:
                return ""
            prefix = f"Строка Excel {row_num}: " if row_num else "Строка: "
            return prefix + " | ".join(parts)

        source_ctx = f"[Документ: {source_name}]\n" if source_name else ""
        ctx        = f"{source_ctx}[Лист: {name}]\nКолонки: {', '.join(headers)}"
        ctx_tokens = _estimate_tokens(ctx)

        sample_lines: list[str] = []
        for row in rows[:5]:
            line = row_to_text(row)
            if line:
                sample_lines.append(line)
            if len(sample_lines) >= 3:
                break

        # Summary chunk: natural-language description so vector search can find
        # this sheet even when the query doesn't match tabular key:value format.
        summary = (
            f"{source_ctx}Таблица Excel. Лист «{name}» содержит {len(rows)} строк. "
            f"Колонки: {', '.join(headers)}."
        )
        if sample_lines:
            summary += "\nПримеры строк:\n" + "\n".join(sample_lines)
        yield {
            "content":     summary,
            "chunk_index": chunk_idx,
            "token_count": _estimate_tokens(summary),
        }
        chunk_idx += 1

        current: list[str] = []
        cur_tok = ctx_tokens

        for row in rows:
            line = row_to_text(row)
            if not line:
                continue
            line_tok = _estimate_tokens(line)

            if current and cur_tok + line_tok > chunk_size:
                yield {
                    "content":     ctx + "\n\n" + "\n".join(current),
                    "chunk_index": chunk_idx,
                    "token_count": cur_tok,
                }
                chunk_idx += 1

                tail     = current[-overlap_rows:] if overlap_rows and line_tok < chunk_size else []
                tail_tok = sum(_estimate_tokens(r) for r in tail)
                current  = tail + [line]
                cur_tok  = ctx_tokens + tail_tok + line_tok
            else:
                current.append(line)
                cur_tok += line_tok

        if current:
            yield {
                "content":     ctx + "\n\n" + "\n".join(current),
                "chunk_index": chunk_idx,
                "token_count": cur_tok,
            }
            chunk_idx += 1


def chunk_pages(
    pages: list[str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Generator[dict, None, None]:
    chunk_idx = 0
    for page_num, page_text in enumerate(pages, 1):
        for chunk in chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
            chunk["page_num"]    = page_num
            chunk["chunk_index"] = chunk_idx
            chunk_idx += 1
            yield chunk
