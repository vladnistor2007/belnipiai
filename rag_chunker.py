import re
from typing import Generator

CHUNK_SIZE = 300
CHUNK_OVERLAP = 60


def _estimate_tokens(text: str) -> int:
    cyrillic = sum(1 for c in text if 'Ѐ' <= c <= 'ӿ')
    return max(1, cyrillic // 2 + (len(text) - cyrillic) // 3)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?;»])\s+|\n{2,}', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> Generator[dict, None, None]:
    sentences = _split_sentences(text)
    if not sentences:
        return

    current: list[str] = []
    current_tokens = 0
    chunk_idx = 0

    for sentence in sentences:
        sent_tokens = _estimate_tokens(sentence)

        if current_tokens + sent_tokens > chunk_size and current:
            yield {
                "content": " ".join(current),
                "chunk_index": chunk_idx,
                "token_count": current_tokens,
            }
            chunk_idx += 1

            # Keep last sentences that fit in the overlap budget
            overlap_tokens = 0
            tail: list[str] = []
            for s in reversed(current):
                t = _estimate_tokens(s)
                if overlap_tokens + t > overlap:
                    break
                tail.insert(0, s)
                overlap_tokens += t

            current = tail
            current_tokens = overlap_tokens

        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        yield {
            "content": " ".join(current),
            "chunk_index": chunk_idx,
            "token_count": current_tokens,
        }


def chunk_pages(pages: list[str], chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> Generator[dict, None, None]:
    chunk_idx = 0
    for page_num, page_text in enumerate(pages, 1):
        for chunk in chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
            chunk["page_num"] = page_num
            chunk["chunk_index"] = chunk_idx
            chunk_idx += 1
            yield chunk
