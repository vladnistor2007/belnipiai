import os
import re
from typing import Optional
import psycopg2
import psycopg2.extras
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

_embed_client = ollama.Client(host=OLLAMA_HOST)


def get_db():
    return psycopg2.connect(DATABASE_URL,
                            cursor_factory=psycopg2.extras.RealDictCursor)


def get_embedding(text: str) -> list[float]:
    resp = _embed_client.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


def estimate_tokens(text: str) -> int:
    cyrillic = sum(1 for c in text if 'Ѐ' <= c <= 'ӿ')
    return max(1, cyrillic // 2 + (len(text) - cyrillic) // 3)


def index_document(document_id: str, chunks: list[dict]) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE kb_documents SET status='indexing' WHERE id=%s",
                (document_id,)
            )
            conn.commit()

            for chunk in chunks:
                embedding = get_embedding(chunk["content"])
                cur.execute("""
                    INSERT INTO doc_chunks
                      (document_id, chunk_index, content, content_tsv,
                       embedding, page_num, token_count)
                    VALUES (%s, %s, %s,
                      to_tsvector('russian', %s),
                      %s::vector, %s, %s)
                """, (
                    document_id,
                    chunk["chunk_index"],
                    chunk["content"],
                    chunk["content"],
                    str(embedding),
                    chunk.get("page_num"),
                    chunk.get("token_count", 0),
                ))

            cur.execute(
                "UPDATE kb_documents "
                "SET status='indexed', chunk_count=%s, updated_at=NOW() "
                "WHERE id=%s",
                (len(chunks), document_id)
            )
        conn.commit()


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


def hybrid_search(query: str, top_k: int = TOP_K,
                  folder_id: Optional[str] = None) -> list[dict]:
    query_vec = get_embedding(query)
    vec_str   = str(query_vec)
    terms     = _keyword_terms(query)

    folder_clause = "AND d.folder_id = %(folder_id)s" if folder_id else ""
    keyword_where = ""
    keyword_score_parts = ["CASE WHEN c.content ILIKE %(phrase)s THEN 1.0 ELSE 0 END"]
    keyword_params = {"phrase": f"%{query}%"}
    if terms:
        term_clauses = []
        for i, term in enumerate(terms):
            key = f"kw{i}"
            keyword_params[key] = f"%{term}%"
            term_clauses.append(f"c.content ILIKE %({key})s")
            keyword_score_parts.append(f"CASE WHEN c.content ILIKE %({key})s THEN 1.0 ELSE 0 END")
        keyword_where = "AND (" + " OR ".join(term_clauses + ["c.content ILIKE %(phrase)s"]) + ")"
    else:
        keyword_where = "AND c.content ILIKE %(phrase)s"
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
        "lim":    max(top_k * 12, CONTEXT_POOL_K),
        "keyword_norm": keyword_norm,
        "folder_id": folder_id,
    }
    params.update(keyword_params)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = 120")
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


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

    results: list[dict] = []
    with get_db() as conn:
        with conn.cursor() as cur:
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
                    cur.execute(
                        """
                        SELECT content, chunk_index, page_num
                        FROM doc_chunks
                        WHERE document_id = %s
                          AND chunk_index BETWEEN %s AND %s
                        ORDER BY chunk_index
                        """,
                        (doc_id, lo, hi),
                    )
                    rows = [dict(r) for r in cur.fetchall()]
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
                         max_tokens: int = 24000) -> list[dict]:
    relevant = expand_with_neighbors(
        hybrid_search(query, top_k=TOP_K, folder_id=folder_id)
    )
    coverage = fetch_document_coverage(folder_id=folder_id)

    combined: list[dict] = []
    combined.extend(relevant)
    combined.extend(coverage)
    return pack_chunks_for_context(combined, max_tokens=max_tokens)


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
