import os
import re
import threading
from collections import OrderedDict
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional
import psycopg2
import psycopg2.extras
import psycopg2.pool
import ollama

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL  = os.environ.get("EMBED_MODEL",  "nomic-embed-text")
DATABASE_URL = os.environ.get("DATABASE_URL",
    "postgresql://postgres:sdd05072008sdd@localhost:5432/belnipiai")

TOP_K           = 80
API_TOP_K       = 30
CONTEXT_POOL_K  = 160
CONTEXT_WINDOW  = 1
TABLE_DOC_TYPES = {"xlsx", "xls", "xlsm", "ods", "csv", "tsv"}
EMBED_BATCH_SIZE = 64
SEARCH_CANDIDATE_MULTIPLIER = 8
HNSW_EF_SEARCH = 96
MAX_PHRASE_KEYWORD_CHARS = 120
MAX_PHRASE_KEYWORD_WORDS = 12
QUERY_EMBED_CACHE_SIZE = 256
SEARCH_CACHE_SIZE = 128
CONTEXT_CACHE_SIZE = 64

_embed_client = ollama.Client(host=OLLAMA_HOST)

_pool: "psycopg2.pool.ThreadedConnectionPool | None" = None
_pool_lock = threading.Lock()
_cache_lock = threading.Lock()
_kb_revision = 0
_query_embed_cache: "OrderedDict[str, list[float]]" = OrderedDict()
_search_cache: "OrderedDict[tuple, list[dict]]" = OrderedDict()
_context_cache: "OrderedDict[tuple, list[dict]]" = OrderedDict()


def _get_pool() -> "psycopg2.pool.ThreadedConnectionPool":
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=10,
                    dsn=DATABASE_URL,
                    cursor_factory=psycopg2.extras.RealDictCursor,
                )
    return _pool


@contextmanager
def get_db():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def get_embedding(text: str) -> list[float]:
    resp = _embed_client.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


def _cache_get(cache: "OrderedDict", key):
    with _cache_lock:
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value


def _cache_put(cache: "OrderedDict", key, value, max_size: int) -> None:
    with _cache_lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)


def _invalidate_search_caches() -> None:
    global _kb_revision
    with _cache_lock:
        _kb_revision += 1
        _search_cache.clear()
        _context_cache.clear()


def invalidate_kb_caches() -> None:
    _invalidate_search_caches()


def _get_query_embedding(text: str) -> list[float]:
    cached = _cache_get(_query_embed_cache, text)
    if cached is not None:
        return list(cached)
    embedding = get_embedding(text)
    _cache_put(_query_embed_cache, text, list(embedding), QUERY_EMBED_CACHE_SIZE)
    return embedding


def _get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch embedding: N texts → 1 HTTP call per EMBED_BATCH_SIZE texts."""
    result: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = _embed_client.embed(model=EMBED_MODEL, input=batch)
        result.extend(resp["embeddings"])
    return result


def estimate_tokens(text: str) -> int:
    cyrillic = sum(1 for c in text if 'Ѐ' <= c <= 'ӿ')
    return max(1, cyrillic // 2 + (len(text) - cyrillic) // 3)


def index_document(document_id: str, chunks: list[dict]) -> None:
    if not chunks:
        return
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE kb_documents SET status='indexing' WHERE id=%s",
                (document_id,)
            )
            conn.commit()

            embeddings = _get_embeddings_batch([c["content"] for c in chunks])

            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO doc_chunks
                  (document_id, chunk_index, content, content_tsv,
                   embedding, page_num, token_count)
                VALUES %s
                """,
                [
                    (
                        document_id,
                        chunk["chunk_index"],
                        chunk["content"],
                        chunk["content"],
                        str(emb),
                        chunk.get("page_num"),
                        chunk.get("token_count", 0),
                    )
                    for chunk, emb in zip(chunks, embeddings)
                ],
                template="(%s, %s, %s, to_tsvector('russian', %s), %s::vector, %s, %s)",
            )

            cur.execute(
                "UPDATE kb_documents "
                "SET status='indexed', chunk_count=%s, updated_at=NOW() "
                "WHERE id=%s",
                (len(chunks), document_id)
            )
        conn.commit()
    _invalidate_search_caches()


def _keyword_terms(query: str, limit: int = 8) -> list[str]:
    terms = []
    for term in re.findall(r"[\w\-./№]+", query, flags=re.UNICODE):
        term = term.strip().lower()
        if len(term) < 2:
            continue
        if term not in terms:
            terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def _use_phrase_keyword(query: str) -> bool:
    if not query:
        return False
    if len(query) > MAX_PHRASE_KEYWORD_CHARS:
        return False
    if len(query.split()) > MAX_PHRASE_KEYWORD_WORDS:
        return False
    return True


def hybrid_search(query: str, top_k: int = TOP_K,
                  folder_id: Optional[str] = None) -> list[dict]:
    cache_key = (_kb_revision, folder_id or "", query, top_k)
    cached = _cache_get(_search_cache, cache_key)
    if cached is not None:
        return [dict(row) for row in cached]

    query_vec = _get_query_embedding(query)
    vec_str   = str(query_vec)
    terms     = _keyword_terms(query)
    use_phrase_keyword = _use_phrase_keyword(query)

    folder_clause = "AND d.folder_id = %(folder_id)s" if folder_id else ""
    keyword_where = ""
    keyword_score_parts = []
    keyword_params = {}
    if use_phrase_keyword:
        keyword_score_parts.append("CASE WHEN c.content ILIKE %(phrase)s THEN 1.0 ELSE 0 END")
        keyword_params["phrase"] = f"%{query}%"
    if terms:
        term_clauses = []
        for i, term in enumerate(terms):
            key = f"kw{i}"
            keyword_params[key] = f"%{term}%"
            term_clauses.append(f"c.content ILIKE %({key})s")
            keyword_score_parts.append(f"CASE WHEN c.content ILIKE %({key})s THEN 1.0 ELSE 0 END")
        keyword_matchers = list(term_clauses)
        if use_phrase_keyword:
            keyword_matchers.append("c.content ILIKE %(phrase)s")
        keyword_where = "AND (" + " OR ".join(keyword_matchers) + ")"
    elif use_phrase_keyword:
        keyword_where = "AND c.content ILIKE %(phrase)s"
    else:
        keyword_where = ""
        keyword_score_parts.append("0.0")
    keyword_score_expr = " + ".join(keyword_score_parts)
    keyword_norm = max(1, len(keyword_score_parts))

    sql = f"""
    WITH vector_part AS (
        SELECT
            c.id,
            ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(vec)s::vector) AS vector_rank,
            1 - (c.embedding <=> %(vec)s::vector) AS vector_score
        FROM doc_chunks c
        JOIN kb_documents d ON d.id = c.document_id
        WHERE d.status = 'indexed'
        {folder_clause}
        ORDER BY c.embedding <=> %(vec)s::vector
        LIMIT %(lim)s
    ),
    text_part AS (
        SELECT
            c.id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(c.content_tsv, plainto_tsquery('russian', %(query)s)) DESC
            ) AS text_rank,
            ts_rank_cd(c.content_tsv,
                       plainto_tsquery('russian', %(query)s)) AS text_score
        FROM doc_chunks c
        JOIN kb_documents d ON d.id = c.document_id
        WHERE d.status = 'indexed'
          AND c.content_tsv @@ plainto_tsquery('russian', %(query)s)
          {folder_clause}
        ORDER BY text_score DESC
        LIMIT %(lim)s
    ),
    keyword_part AS (
        SELECT
            c.id,
            (({keyword_score_expr}) / %(keyword_norm)s::float) AS keyword_score
        FROM doc_chunks c
        JOIN kb_documents d ON d.id = c.document_id
        WHERE d.status = 'indexed'
          {folder_clause}
          {keyword_where}
        ORDER BY keyword_score DESC, c.created_at DESC
        LIMIT %(lim)s
    ),
    candidates AS (
        SELECT id FROM vector_part
        UNION
        SELECT id FROM text_part
        UNION
        SELECT id FROM keyword_part
    )
    SELECT
        c.id,
        c.content,
        c.document_id,
        c.page_num,
        c.chunk_index,
        d.original_name,
        d.doc_type,
        COALESCE(v.vector_score, 0) AS vector_score,
        COALESCE(t.text_score,   0) AS text_score,
        COALESCE(k.keyword_score, 0) AS keyword_score,
        0.45 * COALESCE(1.0 / (60 + v.vector_rank), 0)
          + 0.35 * COALESCE(1.0 / (60 + t.text_rank), 0)
          + 0.20 * COALESCE(k.keyword_score, 0) AS score
    FROM candidates cand
    JOIN doc_chunks c ON c.id = cand.id
    JOIN kb_documents d ON d.id = c.document_id
    LEFT JOIN vector_part v ON v.id = c.id
    LEFT JOIN text_part t ON t.id = c.id
    LEFT JOIN keyword_part k ON k.id = c.id
    ORDER BY score DESC
    LIMIT %(top_k)s
    """

    params = {
        "vec":    vec_str,
        "query":  query,
        "top_k":  top_k,
        "lim":    max(top_k * SEARCH_CANDIDATE_MULTIPLIER, CONTEXT_POOL_K),
        "keyword_norm": keyword_norm,
        "folder_id": folder_id,
    }
    params.update(keyword_params)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = %s", (HNSW_EF_SEARCH,))
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
    _cache_put(_search_cache, cache_key, [dict(row) for row in rows], SEARCH_CACHE_SIZE)
    return rows


def expand_with_neighbors(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return chunks

    doc_ranges: dict[str, list] = {}
    for c in chunks:
        doc_id = c["document_id"]
        base_idx = c["chunk_index"]
        window = 0 if (c.get("doc_type") or "").lower() in TABLE_DOC_TYPES else CONTEXT_WINDOW
        lo = max(0, base_idx - window)
        hi = base_idx + window
        doc_ranges.setdefault(doc_id, []).append(
            [lo, hi, float(c.get("score", 0)), c]
        )

    # Merge overlapping ranges per document (same logic as before)
    all_merged: list[tuple] = []  # (doc_id, lo, hi, score, base)
    for doc_id, ranges in doc_ranges.items():
        ranges.sort(key=lambda x: x[0])
        merged: list[list] = []
        for lo, hi, score, base in ranges:
            if merged and lo <= merged[-1][1] + 1:
                prev = merged[-1]
                prev[1] = max(prev[1], hi)
                if score > prev[2]:
                    prev[2] = score
                    prev[3] = base
            else:
                merged.append([lo, hi, score, base])
        for lo, hi, score, base in merged:
            all_merged.append((doc_id, lo, hi, score, base))

    if not all_merged:
        return []

    # One batch query for all ranges instead of N separate queries
    conditions: list[str] = []
    params: list = []
    for doc_id, lo, hi, _, _ in all_merged:
        conditions.append(
            "(document_id = %s AND chunk_index BETWEEN %s AND %s)"
        )
        params.extend([doc_id, lo, hi])

    sql = (
        "SELECT content, chunk_index, page_num, document_id FROM doc_chunks"
        " WHERE " + " OR ".join(conditions) +
        " ORDER BY document_id, chunk_index"
    )

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            all_rows = [dict(r) for r in cur.fetchall()]

    # Group fetched rows by document_id for O(1) lookup
    doc_rows: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        doc_rows[row["document_id"]].append(row)

    results: list[dict] = []
    for doc_id, lo, hi, score, base in all_merged:
        rows = [r for r in doc_rows[doc_id] if lo <= r["chunk_index"] <= hi]
        if not rows:
            continue
        passage = base.copy()
        passage["content"] = "\n\n".join(r["content"] for r in rows)
        passage["score"] = score
        passage["page_num"] = rows[0]["page_num"]
        results.append(passage)

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


def fetch_document_coverage(folder_id: Optional[str] = None,
                            per_document: int = 12,
                            max_chunks: int = 240) -> list[dict]:
    folder_clause = "AND d.folder_id = %(folder_id)s" if folder_id else ""
    sql = f"""
    WITH ranked AS (
        SELECT
            c.id, c.content, c.document_id, c.page_num, c.chunk_index,
            d.original_name, d.doc_type,
            ROW_NUMBER() OVER (
                PARTITION BY c.document_id ORDER BY c.chunk_index
            ) AS rn
        FROM doc_chunks c
        JOIN kb_documents d ON d.id = c.document_id
        WHERE d.status = 'indexed'
          {folder_clause}
    )
    SELECT
        id, content, document_id, page_num, chunk_index,
        original_name, doc_type,
        0.0 AS vector_score,
        0.0 AS text_score,
        0.0 AS keyword_score,
        0.001 AS score
    FROM ranked
    WHERE rn <= %(per_document)s
    ORDER BY rn, original_name, chunk_index
    LIMIT %(max_chunks)s
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "folder_id": folder_id,
                "per_document": per_document,
                "max_chunks": max_chunks,
            })
            return [dict(r) for r in cur.fetchall()]


def fetch_all_context_pages(folder_id: Optional[str] = None,
                            max_chunks: int = 2000) -> tuple[list[str], int]:
    folder_clause = "AND d.folder_id = %(folder_id)s" if folder_id else ""
    count_sql = f"""
        SELECT COUNT(*) AS cnt
        FROM doc_chunks c
        JOIN kb_documents d ON d.id = c.document_id
        WHERE d.status = 'indexed'
          {folder_clause}
    """
    data_sql = f"""
        SELECT c.content, c.page_num, c.chunk_index, d.original_name
        FROM doc_chunks c
        JOIN kb_documents d ON d.id = c.document_id
        WHERE d.status = 'indexed'
          {folder_clause}
        ORDER BY d.original_name, c.chunk_index
        LIMIT %(max_chunks)s
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            params = {"folder_id": folder_id, "max_chunks": max_chunks}
            cur.execute(count_sql, params)
            total = int(cur.fetchone()["cnt"] or 0)
            cur.execute(data_sql, params)
            rows = [dict(r) for r in cur.fetchall()]

    pages = []
    for row in rows:
        page = f", стр. {row['page_num']}" if row.get("page_num") else ""
        pages.append(
            f"[Источник: {row['original_name']}{page}, чанк {row['chunk_index']}]\n"
            f"{row['content']}"
        )
    return pages, total


def pack_chunks_for_context(chunks: list[dict], max_tokens: int) -> list[dict]:
    packed: list[dict] = []
    seen_ids = set()
    used = 0

    for chunk in chunks:
        chunk_id = chunk.get("id")
        key = chunk_id or (
            chunk.get("document_id"),
            chunk.get("chunk_index"),
            chunk.get("content", "")[:120],
        )
        if key in seen_ids:
            continue
        text = chunk.get("content") or ""
        cost = estimate_tokens(text) + 60
        if packed and used + cost > max_tokens:
            continue
        packed.append(chunk)
        seen_ids.add(key)
        used += cost
        if used >= max_tokens:
            break

    packed.sort(
        key=lambda c: (
            c.get("original_name") or "",
            c.get("chunk_index", 0),
            -float(c.get("score", 0) or 0),
        )
    )
    return packed


def build_context_chunks(query: str, folder_id: Optional[str] = None,
                         max_tokens: int = 24000,
                         include_coverage: bool = False) -> list[dict]:
    cache_key = (_kb_revision, folder_id or "", query, max_tokens, include_coverage)
    cached = _cache_get(_context_cache, cache_key)
    if cached is not None:
        return [dict(row) for row in cached]

    relevant = expand_with_neighbors(
        hybrid_search(query, top_k=TOP_K, folder_id=folder_id)
    )

    combined: list[dict] = []
    combined.extend(relevant)
    if include_coverage:
        combined.extend(fetch_document_coverage(folder_id=folder_id))
    packed = pack_chunks_for_context(combined, max_tokens=max_tokens)
    _cache_put(_context_cache, cache_key, [dict(row) for row in packed], CONTEXT_CACHE_SIZE)
    return packed


def build_rag_context(chunks: list[dict], max_tokens: Optional[int] = None) -> str:
    parts = []
    used = 0
    for i, chunk in enumerate(chunks, 1):
        page = f", стр. {chunk['page_num']}" if chunk.get("page_num") else ""
        part = f"[Источник {i}: {chunk['original_name']}{page}]\n{chunk['content']}"
        cost = estimate_tokens(part)
        if max_tokens is not None and parts and used + cost > max_tokens:
            continue
        parts.append(part)
        used += cost
    return "\n\n---\n\n".join(parts)
