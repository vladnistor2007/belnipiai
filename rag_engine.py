import os
from typing import Optional
import psycopg2
import psycopg2.extras
import ollama

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL  = os.environ.get("EMBED_MODEL",  "nomic-embed-text")
DATABASE_URL = os.environ.get("DATABASE_URL",
    "postgresql://postgres:sdd05072008sdd@localhost:5432/belnipiai")

TOP_K        = 10
HYBRID_ALPHA = 0.6   # доля векторного поиска; (1-α) — доля BM25/tsvector

_embed_client = ollama.Client(host=OLLAMA_HOST)


def get_db():
    return psycopg2.connect(DATABASE_URL,
                            cursor_factory=psycopg2.extras.RealDictCursor)


def get_embedding(text: str) -> list[float]:
    resp = _embed_client.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


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


def hybrid_search(query: str, top_k: int = TOP_K,
                  folder_id: Optional[str] = None) -> list[dict]:
    query_vec = get_embedding(query)
    vec_str   = str(query_vec)

    folder_clause = "AND d.folder_id = %(folder_id)s" if folder_id else ""

    sql = f"""
    WITH vector_part AS (
        SELECT
            c.id, c.content, c.document_id, c.page_num, c.chunk_index,
            d.original_name,
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
            ts_rank(c.content_tsv,
                    plainto_tsquery('russian', %(query)s)) AS text_score
        FROM doc_chunks c
        JOIN kb_documents d ON d.id = c.document_id
        WHERE d.status = 'indexed'
          AND c.content_tsv @@ plainto_tsquery('russian', %(query)s)
          {folder_clause}
        LIMIT %(lim)s
    )
    SELECT
        v.id, v.content, v.document_id, v.page_num, v.chunk_index,
        v.original_name, v.vector_score,
        COALESCE(t.text_score, 0) AS text_score,
        %(alpha)s * v.vector_score
          + (1 - %(alpha)s) * COALESCE(t.text_score, 0) AS score
    FROM vector_part v
    LEFT JOIN text_part t ON t.id = v.id
    ORDER BY score DESC
    LIMIT %(top_k)s
    """

    params = {
        "vec":    vec_str,
        "query":  query,
        "top_k":  top_k,
        "lim":    top_k * 3,
        "alpha":  HYBRID_ALPHA,
        "folder_id": folder_id,
    }

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def build_rag_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page = f", стр. {chunk['page_num']}" if chunk.get("page_num") else ""
        parts.append(
            f"[Источник {i}: {chunk['original_name']}{page}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)
