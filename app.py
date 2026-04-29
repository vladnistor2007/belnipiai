import os
import io
import re
import base64
import json
import time
import hmac
import hashlib
import tempfile
import uuid
import threading
import urllib.request
from flask import Flask, request, Response, send_from_directory, stream_with_context, session, redirect, jsonify
import psycopg2
import psycopg2.extras
import ollama
from rag_engine import (
    index_document as rag_index_document,
    hybrid_search,
    build_rag_context,
)
from rag_chunker import chunk_text as rag_chunk_text, chunk_pages as rag_chunk_pages

app = Flask(__name__, static_folder="static")
ROOT_DIR = app.root_path
app.secret_key = os.environ.get("AI_FLASK_SECRET", "change-this-ai-flask-secret")
app.config.update(
    SESSION_COOKIE_NAME='belnipiai_session',
    SESSION_COOKIE_PATH='/',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)

MODEL_DEFAULT = os.environ.get("MODEL_DEFAULT", "gemma3:4b")
AI_SSO_SHARED_SECRET = os.environ.get("AI_SSO_SHARED_SECRET", "change-this-ai-sso-secret")
AI_SSO_MAX_AGE_SEC = int(os.environ.get("AI_SSO_MAX_AGE_SEC", "300"))

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:sdd05072008sdd@localhost:5432/belnipiai")

_client = ollama.Client(host=OLLAMA_HOST)


def _current_user() -> str:
    return _normalize_username(session.get("username") or "")


def _normalize_username(raw_username: str) -> str:
    login = (raw_username or "").strip().lower()
    if "\\" in login:
        login = login.split("\\")[-1].strip()
    if "@" in login:
        login = login.split("@")[0].strip()
    return login


def _verify_sso_payload(username: str, display_name: str, ts_raw: str, sig: str) -> bool:
    try:
        ts_val = int(ts_raw)
    except Exception:
        return False
    if abs(int(time.time()) - ts_val) > AI_SSO_MAX_AGE_SEC:
        return False
    payload = f"{username}|{display_name}|{ts_raw}"
    expected = hmac.new(
        AI_SSO_SHARED_SECRET.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig or "")


# ── модели ────────────────────────────────────────────────────────────────────

# Отображаемые названия для фронтенда
MODEL_LABELS = {
    "gemma3:4b":  "Базовая",   # ноутбук
    "gemma4:e4b": "Базовая",   # сервер
    "gemma4:26b": "Pro",        # сервер
}

MODEL_PROFILES = {
    "gemma3:4b":  {"num_ctx": 16000, "chunk_tokens": 12000, "no_think": False},
    "gemma4:e4b": {"num_ctx": 32000, "chunk_tokens": 26000, "no_think": False},
    "gemma4:26b": {"num_ctx": 32000, "chunk_tokens": 26000, "no_think": False},
    "qwen3:14b":  {"num_ctx": 32000, "chunk_tokens": 26000, "no_think": False},
    "qwen3":      {"num_ctx": 32000, "chunk_tokens": 26000, "no_think": False},
}

DEFAULT_PROFILE = {"num_ctx": 32000, "chunk_tokens": 26000, "no_think": False}

MODEL_TASK_PRIORITY = {
    "image":    (["gemma4:26b", "gemma4:e4b", "gemma3:4b"], "Изображение — нужна vision-модель"),
    "document": (["gemma4:26b", "gemma4:e4b", "gemma3:4b"], "Документ — нужен большой контекст"),
    "table":    (["gemma4:26b", "gemma4:e4b", "gemma3:4b"], "Таблица — нужна аналитическая модель"),
    "complex":  (["gemma4:26b", "gemma4:e4b", "gemma3:4b"], "Сложный вопрос — нужна умная модель"),
    "simple":   (["gemma3:4b", "gemma4:e4b", "gemma4:26b"], "Простой вопрос"),
}

COMPLEX_KEYWORDS = {
    "проанализируй", "сравни", "объясни", "почему", "как работает",
    "разбери", "оцени", "составь", "напиши", "рассчитай", "найди противоречия",
}


def suggest_model(file_ext: str, question: str, available: list) -> tuple[str, str]:
    """Возвращает (model_name, reason) из доступных моделей."""

    def pick(task_key: str) -> str | None:
        preferred, _ = MODEL_TASK_PRIORITY[task_key]
        for p in preferred:
            base = p.split(":")[0]
            for m in available:
                if m == p or m.startswith(base):
                    return m
        return None

    images = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
    docs   = {".pdf", ".docx", ".doc"}
    tables = {".xlsx", ".xls", ".xlsm", ".ods", ".csv", ".tsv"}

    if file_ext in images:
        task = "image"
    elif file_ext in docs:
        task = "document"
    elif file_ext in tables:
        task = "table"
    elif len(question) > 180 or any(kw in question.lower() for kw in COMPLEX_KEYWORDS):
        task = "complex"
    else:
        task = "simple"

    model = pick(task) or (available[0] if available else MODEL_DEFAULT)
    _, reason = MODEL_TASK_PRIORITY[task]
    return model, reason


SYSTEM_PROMPT_DOC = (
    "You are a precise document analyst. "
    "Answer ONLY based on the provided document. "
    "If the answer is not found in the document, say so explicitly — never invent facts. "
    "Format your response in Markdown."
)

SYSTEM_PROMPT_CHAT = (
    "You are a helpful assistant. "
    "If you are not certain about something, say so — never make up facts, numbers or dates. "
    "Format your responses in Markdown."
)

SYSTEM_PROMPT_RAG = (
    "You are an expert assistant working with enterprise documents. "
    "Answer ONLY based on the provided context retrieved from the knowledge base. "
    "For every fact or claim, cite the source like this: [Источник N]. "
    "If the answer cannot be found in the context, respond exactly: "
    "'В базе знаний нет информации по этому вопросу.' "
    "Never invent facts. Format your response in Markdown."
)


def get_profile(model: str) -> dict:
    if model in MODEL_PROFILES:
        return MODEL_PROFILES[model]
    for key, profile in MODEL_PROFILES.items():
        if model.startswith(key.split(":")[0]):
            return profile
    return DEFAULT_PROFILE


def apply_no_think(text: str, model: str) -> str:
    if get_profile(model)["no_think"]:
        return "/no_think " + text
    return text


def estimate_tokens(text: str) -> int:
    """Быстрая локальная оценка без сетевого вызова (для решений о чанкинге)."""
    cyrillic = sum(1 for c in text if 'Ѐ' <= c <= 'ӿ')
    return max(1, cyrillic // 2 + (len(text) - cyrillic) // 3)


def count_tokens(text: str, model: str = MODEL_DEFAULT) -> int:
    """Точный подсчёт через Ollama /api/tokenize, fallback на estimate_tokens."""
    try:
        body = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/tokenize",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return max(1, len(json.loads(resp.read()).get("tokens", [])))
    except Exception:
        return estimate_tokens(text)


# ── база данных ───────────────────────────────────────────────────────────────

def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    schema = os.path.join(ROOT_DIR, "schema.sql")
    if not os.path.exists(schema):
        return
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(open(schema).read())
        conn.commit()


# ── управление контекстом ─────────────────────────────────────────────────────

def load_history(conv_id: str) -> list:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM messages "
                "WHERE conversation_id = %s AND role != 'summary' "
                "ORDER BY created_at",
                (conv_id,)
            )
            rows = cur.fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def load_history_with_summary(conv_id: str, num_ctx: int) -> list:
    """Загружает историю. Для сообщений с файлами восстанавливает документ из БД."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content, created_at FROM messages "
                "WHERE conversation_id = %s AND role = 'summary' "
                "ORDER BY created_at DESC LIMIT 1",
                (conv_id,)
            )
            summary_row = cur.fetchone()

            since = summary_row["created_at"] if summary_row else None
            base_q = """
                SELECT m.id, m.role, m.content, m.token_count,
                       a.extracted_text, a.original_name
                FROM messages m
                LEFT JOIN attachments a ON a.message_id = m.id
                WHERE m.conversation_id = %s AND m.role != 'summary'
            """
            if since:
                cur.execute(base_q + " AND m.created_at > %s ORDER BY m.created_at", (conv_id, since))
            else:
                cur.execute(base_q + " ORDER BY m.created_at", (conv_id,))
            rows = cur.fetchall()

    rows = list(rows)

    # Только последний документ получает полный текст в контекст,
    # старые документы — лишь ссылка на имя файла
    max_doc_chars = int(num_ctx * 0.55) * 4
    last_doc_idx = max(
        (i for i, r in enumerate(rows) if r["role"] == "user" and r["extracted_text"]),
        default=None
    )

    def build_content(row, idx):
        if row["role"] == "user" and row["extracted_text"]:
            if idx == last_doc_idx:
                doc = row["extracted_text"]
                if len(doc) > max_doc_chars:
                    doc = doc[:max_doc_chars] + "\n\n[... документ обрезан из-за лимита контекста ...]"
                return f"<document name=\"{row['original_name']}\">\n{doc}\n</document>\n\nВопрос: {row['content']}"
            return f"[Ранее загружен документ: {row['original_name']}]\n\nВопрос: {row['content']}"
        return row["content"]

    messages = []
    if summary_row:
        messages.append({"role": "user", "content": f"[Резюме предыдущего разговора]\n{summary_row['content']}"})
        messages.append({"role": "assistant", "content": "Понял, продолжаю разговор с учётом предыдущего контекста."})

    # Считаем токены по фактическому контенту (с документом), а не по r["content"].
    # Это правильно при смене модели: сохранённый token_count от старой модели не используется.
    total_tokens = sum(estimate_tokens(build_content(r, i)) for i, r in enumerate(rows))
    threshold = int(num_ctx * 0.85)

    if total_tokens <= threshold or len(rows) <= 4:
        messages += [{"role": r["role"], "content": build_content(r, i)} for i, r in enumerate(rows)]
        return messages

    # Нужна обрезка. Бюджет = threshold минус уже добавленные summary-сообщения.
    budget = threshold - sum(estimate_tokens(m["content"]) for m in messages)

    # Якоря: последний документ + ответ на него — всегда включаем, независимо от бюджета.
    anchors: set[int] = set()
    if last_doc_idx is not None:
        anchors.add(last_doc_idx)
        nxt = last_doc_idx + 1
        if nxt < len(rows) and rows[nxt]["role"] == "assistant":
            anchors.add(nxt)

    for idx in sorted(anchors):
        budget -= estimate_tokens(build_content(rows[idx], idx))

    # Добираем свежие сообщения (от конца) в рамках оставшегося бюджета.
    tail: set[int] = set()
    for i in range(len(rows) - 1, -1, -1):
        if i in anchors:
            continue
        t = estimate_tokens(rows[i]["content"])
        if budget - t < 100:
            break
        tail.add(i)
        budget -= t

    keep = sorted(anchors | tail)
    messages += [{"role": rows[i]["role"], "content": build_content(rows[i], i)} for i in keep]

    return messages


def maybe_summarize(conv_id: str, model: str, num_ctx: int):
    """Если история занимает > 65% контекста — делаем резюме (вызывается из фонового треда)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT SUM(token_count) FROM messages "
                "WHERE conversation_id = %s AND role != 'summary'",
                (conv_id,)
            )
            total = cur.fetchone()["sum"] or 0

    if total < int(num_ctx * 0.65):
        return

    history = load_history(conv_id)
    if len(history) < 6:
        return

    old_messages = history[:-4]
    text_to_summarize = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in old_messages)

    resp = _client.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Сделай краткое резюме этого диалога (важные факты, решения, контекст):\n\n{text_to_summarize}"
        }],
        options={"num_ctx": num_ctx, "temperature": 0.1},
        stream=False
    )
    summary_text = resp["message"]["content"]

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content, token_count) VALUES (%s, 'summary', %s, %s)",
                (conv_id, summary_text, count_tokens(summary_text, model))
            )
        conn.commit()


def save_message(conv_id: str, role: str, content: str, model: str = MODEL_DEFAULT,
                 attachment_name: str = None, attachment_text: str = None):
    tokens = count_tokens(content, model)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (id, conversation_id, role, content, token_count) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (str(uuid.uuid4()), conv_id, role, content, tokens)
            )
            msg_id = cur.fetchone()["id"]
            if attachment_name and attachment_text:
                cur.execute(
                    "INSERT INTO attachments (message_id, original_name, extracted_text) VALUES (%s, %s, %s)",
                    (msg_id, attachment_name, attachment_text)
                )
            cur.execute(
                "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
                (conv_id,)
            )
        conn.commit()


def generate_title(question: str, model: str) -> str:
    try:
        resp = _client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты генерируешь названия чатов. "
                        "Отвечай ТОЛЬКО самим названием — одной строкой, 2–4 слова. "
                        "Никаких объяснений, никаких списков, никаких кавычек."
                    )
                },
                {
                    "role": "user",
                    "content": f"Название чата по запросу:\n{question[:400]}"
                }
            ],
            options={"num_ctx": 2048, "temperature": 0.1},
            stream=False
        )
        raw = resp["message"]["content"]
        # Убираем теги <think>...</think>
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Если модель вернула список — берём первый пункт
        m = re.search(r"^[\d\-\*•]\s*\.?\s*(.+)$", raw, re.MULTILINE)
        title = m.group(1).strip() if m else next(
            (ln.strip() for ln in raw.splitlines() if ln.strip()), ""
        )
        # Убираем типичные «мусорные» префиксы
        title = re.sub(
            r"(?i)^(название|title|тема|чат|вот|итого|ответ)\s*[:\-–—]?\s*", "", title
        ).strip()
        title = title.strip('"\'«»').rstrip(".,;:!?").strip()
        return title[:60] if len(title) > 2 else question[:50]
    except Exception:
        return question[:50]


# ── утилиты файлов ────────────────────────────────────────────────────────────

def to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG", optimize=False, compress_level=0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_available_models():
    try:
        result = _client.list()
        return [m.model for m in result.models]
    except Exception:
        return [MODEL_DEFAULT]


def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def detect_pdf_type(path):
    import pymupdf
    doc = pymupdf.open(path)
    total_chars = sum(len(p.get_text().strip()) for p in doc)
    avg = total_chars / max(len(doc), 1)
    return "scanned" if avg < 50 else "text"


def read_pdf_text(path):
    import pymupdf
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"[Страница {i+1}]\n{text}")
    return pages


def read_docx(path):
    import docx
    doc = docx.Document(path)
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if row_text:
                parts.append(row_text)
    return "\n".join(parts)


def read_excel(path):
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    xf = pd.ExcelFile(path, engine="xlrd" if ext == ".xls" else "openpyxl")
    parts = []
    for sheet in xf.sheet_names:
        df = xf.parse(sheet)
        parts.append(f"[Лист: {sheet}]\n{df.to_string()}")
    return "\n\n".join(parts)


def read_csv(path, ext=".csv"):
    import pandas as pd
    sep = "\t" if ext == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep, encoding="utf-8", errors="ignore")
    except Exception:
        df = pd.read_csv(path, sep=sep, encoding="cp1251", errors="ignore")
    return df.to_string()


# ── стриминг ──────────────────────────────────────────────────────────────────

def stream_text_response(content, question, model, history=None):
    profile = get_profile(model)
    num_ctx = profile["num_ctx"]

    max_chars = (num_ctx - 1000) * 4
    if len(content) > max_chars:
        content = content[:max_chars]

    messages = [{"role": "system", "content": SYSTEM_PROMPT_DOC}]
    if history:
        messages += history
    messages.append({
        "role": "user",
        "content": apply_no_think(f"<document>\n{content}\n</document>\n\nВопрос: {question}", model)
    })

    response = _client.chat(model=model, messages=messages,
                            options={"num_ctx": num_ctx, "temperature": 0.1}, stream=True)
    for chunk in response:
        yield f"data: {json.dumps({'text': chunk['message']['content']})}\n\n"
    yield "data: [DONE]\n\n"


def stream_image_response(images_b64, question, model, history=None):
    profile = get_profile(model)
    messages = []
    if history:
        messages += history
    messages.append({
        "role": "user",
        "content": apply_no_think(question, model),
        "images": images_b64
    })
    response = _client.chat(model=model, messages=messages,
                            options={"num_ctx": profile["num_ctx"], "temperature": 0.1}, stream=True)
    for chunk in response:
        yield f"data: {json.dumps({'text': chunk['message']['content']})}\n\n"
    yield "data: [DONE]\n\n"


def stream_map_reduce(pages_text, question, model, history=None):
    profile = get_profile(model)
    num_ctx = profile["num_ctx"]
    chunk_chars = profile["chunk_tokens"] * 4

    chunks, current, cur_len = [], [], 0
    for page in pages_text:
        if cur_len + len(page) > chunk_chars and current:
            chunks.append("\n\n".join(current))
            current, cur_len = [page], len(page)
        else:
            current.append(page)
            cur_len += len(page)
    if current:
        chunks.append("\n\n".join(current))

    total = len(chunks)
    yield f"data: {json.dumps({'text': f'📦 Документ большой — обрабатываю {total} частей...\\n\\n'})}\n\n"

    summaries, prev_context = [], ""
    for i, chunk in enumerate(chunks):
        yield f"data: {json.dumps({'text': f'⏳ Часть {i+1}/{total}...\\n'})}\n\n"
        context_hint = f"\n\nКонтекст из предыдущих частей:\n{prev_context}\n" if prev_context else ""
        prompt = apply_no_think(
            f"Это часть {i+1} из {total} документа.\nВопрос: {question}{context_hint}\n"
            f"Извлеки всё относящееся к вопросу, сохрани детали, цифры, имена:\n\n{chunk}", model
        )
        s = ""
        for chunk_r in _client.chat(model=model,
                                     messages=[{"role": "user", "content": prompt}],
                                     options={"num_ctx": num_ctx, "temperature": 0.1}, stream=True):
            s += chunk_r["message"]["content"]
        summaries.append(f"[Часть {i+1}/{total}]\n{s}")
        prev_context = s[-500:] if len(s) > 500 else s

    yield f"data: {json.dumps({'text': '\\n---\\n📝 **Финальный ответ**\\n\\n'})}\n\n"
    yield from stream_text_response("\n\n".join(summaries), question, model, history)


# ── роуты: конверсации ────────────────────────────────────────────────────────

@app.route("/api/conversations", methods=["GET"])
def api_list_conversations():
    username = _current_user()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, model, updated_at FROM conversations "
                "WHERE username = %s ORDER BY updated_at DESC",
                (username,)
            )
            rows = cur.fetchall()
    return {"conversations": [dict(r) for r in rows]}


@app.route("/api/conversations", methods=["POST"])
def api_create_conversation():
    data = request.get_json(force=True)
    model = data.get("model", MODEL_DEFAULT)
    cid = str(uuid.uuid4())
    username = _current_user()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (id, username, title, model) VALUES (%s, %s, %s, %s)",
                (cid, username, "Новый чат", model)
            )
        conn.commit()
    return {"id": cid, "title": "Новый чат", "model": model}


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def api_delete_conversation(conv_id):
    username = _current_user()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE id = %s AND username = %s",
                (conv_id, username)
            )
        conn.commit()
    return {"ok": True}


@app.route("/api/conversations/<conv_id>/messages", methods=["GET"])
def api_get_messages(conv_id):
    username = _current_user()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT c.id FROM conversations c WHERE c.id = %s AND c.username = %s",
                (conv_id, username)
            )
            if not cur.fetchone():
                return {"messages": []}
            cur.execute(
                "SELECT m.id, m.role, m.content, m.created_at, "
                "a.original_name as file_name "
                "FROM messages m "
                "LEFT JOIN attachments a ON a.message_id = m.id "
                "WHERE m.conversation_id = %s AND m.role != 'summary' "
                "ORDER BY m.created_at",
                (conv_id,)
            )
            rows = cur.fetchall()
    return {"messages": [dict(r) for r in rows]}


@app.route("/api/conversations/<conv_id>/title", methods=["PATCH"])
def api_update_title(conv_id):
    data = request.get_json(force=True)
    title = data.get("title", "")[:80]
    username = _current_user()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET title = %s WHERE id = %s AND username = %s",
                (title, conv_id, username)
            )
        conn.commit()
    return {"ok": True}


# ── роут: чат ─────────────────────────────────────────────────────────────────

@app.route("/api/conversations/<conv_id>/chat", methods=["POST"])
def api_chat(conv_id):
    question  = request.form.get("question", "").strip()
    model     = request.form.get("model", MODEL_DEFAULT)
    file      = request.files.get("file")
    use_rag   = request.form.get("use_rag", "false").lower() == "true"
    folder_id = request.form.get("folder_id") or None

    if not question:
        return {"error": "Вопрос не может быть пустым"}, 400

    username = _current_user()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM conversations WHERE id = %s AND username = %s",
                (conv_id, username)
            )
            if not cur.fetchone():
                return {"error": "Чат не найден"}, 404

    profile = get_profile(model)

    # Сохраняем файл до генератора
    tmp_path = file_name = extracted_text = None
    ext = ""
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        file_name = file.filename
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp_path = tmp.name
        tmp.close()
        file.save(tmp_path)

    # Загружаем историю для контекста
    history = load_history_with_summary(conv_id, profile["num_ctx"])

    # Сохраняем вопрос пользователя
    save_message(conv_id, "user", question, model)

    def generate():
        nonlocal extracted_text
        full_response = []

        try:
            if not tmp_path:
                if use_rag:
                    rag_chunks = hybrid_search(question, top_k=10, folder_id=folder_id)
                    if rag_chunks:
                        rag_context = build_rag_context(rag_chunks)
                        sources = [
                            {
                                "name":  c["original_name"],
                                "page":  c.get("page_num"),
                                "score": round(float(c["score"]), 3),
                            }
                            for c in rag_chunks
                        ]
                        yield f"data: {json.dumps({'sources': sources})}\n\n"
                        messages = [{"role": "system", "content": SYSTEM_PROMPT_RAG}]
                        messages += history
                        messages.append({
                            "role": "user",
                            "content": (
                                f"<context>\n{rag_context}\n</context>\n\n"
                                f"Вопрос: {question}"
                            ),
                        })
                        temp = 0.1
                    else:
                        yield f"data: {json.dumps({'text': '_В базе знаний не найдено релевантных документов. Отвечаю из общих знаний._\\n\\n'})}\n\n"
                        messages = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
                        messages += history
                        messages.append({"role": "user", "content": apply_no_think(question, model)})
                        temp = 0.7
                else:
                    messages = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
                    messages += history
                    messages.append({"role": "user", "content": apply_no_think(question, model)})
                    temp = 0.7

                for chunk in _client.chat(model=model, messages=messages,
                                          options={"num_ctx": profile["num_ctx"], "temperature": temp},
                                          stream=True):
                    delta = chunk["message"]["content"]
                    full_response.append(delta)
                    yield f"data: {json.dumps({'text': delta})}\n\n"
                yield "data: [DONE]\n\n"

            else:
                if ext in (".txt", ".md", ".log"):
                    text = read_txt(tmp_path)
                    extracted_text = text
                    gen = (stream_text_response(text, question, model, history)
                           if estimate_tokens(text) <= profile["chunk_tokens"]
                           else stream_map_reduce(
                               [text[i:i+profile["chunk_tokens"]*4]
                                for i in range(0, len(text), profile["chunk_tokens"]*4)],
                               question, model, history))
                    for chunk in gen:
                        if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
                            try:
                                full_response.append(json.loads(chunk[6:]).get("text", ""))
                            except Exception:
                                pass
                        yield chunk

                elif ext == ".pdf":
                    pdf_type = detect_pdf_type(tmp_path)
                    if pdf_type == "text":
                        pages = read_pdf_text(tmp_path)
                        extracted_text = "\n\n".join(pages)
                        gen = (stream_text_response(extracted_text, question, model, history)
                               if estimate_tokens(extracted_text) <= profile["chunk_tokens"]
                               else stream_map_reduce(pages, question, model, history))
                        for chunk in gen:
                            if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
                                try:
                                    full_response.append(json.loads(chunk[6:]).get("text", ""))
                                except Exception:
                                    pass
                            yield chunk
                    else:
                        yield f"data: {json.dumps({'text': '🖼️ Сканированный PDF — конвертирую...\\n\\n'})}\n\n"
                        from pdf2image import convert_from_path
                        pil_pages = convert_from_path(tmp_path, dpi=200)
                        if len(pil_pages) <= 5:
                            images_b64 = [to_base64(p) for p in pil_pages]
                            for chunk in stream_image_response(images_b64, question, model, history):
                                if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
                                    try:
                                        full_response.append(json.loads(chunk[6:]).get("text", ""))
                                    except Exception:
                                        pass
                                yield chunk
                        else:
                            page_texts = []
                            for i, pil_page in enumerate(pil_pages):
                                yield f"data: {json.dumps({'text': f'🔍 Страница {i+1}/{len(pil_pages)}...\\n'})}\n\n"
                                b64 = to_base64(pil_page)
                                r = _client.chat(
                                    model=model,
                                    messages=[{"role": "user",
                                               "content": apply_no_think(f"Страница {i+1} из {len(pil_pages)}. Извлеки весь текст.", model),
                                               "images": [b64]}],
                                    options={"num_ctx": profile["num_ctx"], "temperature": 0.1},
                                    stream=False
                                )
                                page_texts.append(f"[Страница {i+1}]\n{r['message']['content']}")
                            extracted_text = "\n\n".join(page_texts)
                            for chunk in stream_map_reduce(page_texts, question, model, history):
                                if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
                                    try:
                                        full_response.append(json.loads(chunk[6:]).get("text", ""))
                                    except Exception:
                                        pass
                                yield chunk

                elif ext in (".docx", ".doc"):
                    text = read_docx(tmp_path)
                    extracted_text = text
                    gen = (stream_text_response(text, question, model, history)
                           if estimate_tokens(text) <= profile["chunk_tokens"]
                           else stream_map_reduce(
                               [text[i:i+profile["chunk_tokens"]*4]
                                for i in range(0, len(text), profile["chunk_tokens"]*4)],
                               question, model, history))
                    for chunk in gen:
                        if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
                            try:
                                full_response.append(json.loads(chunk[6:]).get("text", ""))
                            except Exception:
                                pass
                        yield chunk

                elif ext in (".xlsx", ".xls", ".xlsm", ".ods"):
                    text = read_excel(tmp_path)
                    extracted_text = text
                    for chunk in stream_text_response(text, question, model, history):
                        if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
                            try:
                                full_response.append(json.loads(chunk[6:]).get("text", ""))
                            except Exception:
                                pass
                        yield chunk

                elif ext in (".csv", ".tsv"):
                    text = read_csv(tmp_path, ext)
                    extracted_text = text
                    for chunk in stream_text_response(text, question, model, history):
                        if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
                            try:
                                full_response.append(json.loads(chunk[6:]).get("text", ""))
                            except Exception:
                                pass
                        yield chunk

                elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"):
                    with open(tmp_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    for chunk in stream_image_response([b64], question, model, history):
                        if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
                            try:
                                full_response.append(json.loads(chunk[6:]).get("text", ""))
                            except Exception:
                                pass
                        yield chunk
                else:
                    yield f"data: {json.dumps({'error': f'Формат {ext} не поддерживается'})}\n\n"

        finally:
            if tmp_path:
                os.unlink(tmp_path)

            # Сохраняем ответ ассистента
            assistant_text = "".join(full_response)
            if assistant_text:
                save_message(conv_id, "assistant", assistant_text, model,
                             attachment_name=file_name,
                             attachment_text=extracted_text)

                # Генерируем название чата после первого ответа
                with get_db() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = %s AND role = 'user'",
                            (conv_id,)
                        )
                        cnt = cur.fetchone()["cnt"]

                if cnt == 1:
                    title = generate_title(question, model)
                    with get_db() as conn:
                        with conn.cursor() as cur:
                            cur.execute("UPDATE conversations SET title = %s WHERE id = %s", (title, conv_id))
                        conn.commit()
                    yield f"data: {json.dumps({'title': title, 'conv_id': conv_id})}\n\n"

                threading.Thread(
                    target=maybe_summarize,
                    args=(conv_id, model, profile["num_ctx"]),
                    daemon=True
                ).start()

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ── База знаний: папки ────────────────────────────────────────────────────────

@app.route("/api/kb/folders", methods=["GET"])
def api_kb_folders():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, description FROM doc_folders ORDER BY name")
            return {"folders": [dict(r) for r in cur.fetchall()]}


@app.route("/api/kb/folders", methods=["POST"])
def api_kb_create_folder():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return {"error": "Имя папки не может быть пустым"}, 400
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO doc_folders (name, description) VALUES (%s, %s) RETURNING id, name",
                (name, data.get("description", ""))
            )
            row = cur.fetchone()
        conn.commit()
    return dict(row), 201


@app.route("/api/kb/folders/<folder_id>", methods=["DELETE"])
def api_kb_delete_folder(folder_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM doc_folders WHERE id=%s", (folder_id,))
        conn.commit()
    return {"ok": True}


# ── База знаний: документы ────────────────────────────────────────────────────

@app.route("/api/kb/documents", methods=["GET"])
def api_kb_documents():
    folder_id = request.args.get("folder_id")
    with get_db() as conn:
        with conn.cursor() as cur:
            if folder_id:
                cur.execute(
                    "SELECT id, original_name, status, chunk_count, created_at, meta "
                    "FROM kb_documents WHERE folder_id=%s ORDER BY created_at DESC",
                    (folder_id,)
                )
            else:
                cur.execute(
                    "SELECT id, original_name, status, chunk_count, created_at, meta "
                    "FROM kb_documents ORDER BY created_at DESC"
                )
            return {"documents": [dict(r) for r in cur.fetchall()]}


@app.route("/api/kb/documents/<doc_id>", methods=["GET"])
def api_kb_document_status(doc_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, original_name, status, chunk_count, meta FROM kb_documents WHERE id=%s",
                (doc_id,)
            )
            row = cur.fetchone()
    if not row:
        return {"error": "Документ не найден"}, 404
    return dict(row)


@app.route("/api/kb/upload", methods=["POST"])
def api_kb_upload():
    file      = request.files.get("file")
    folder_id = request.form.get("folder_id") or None

    if not file or not file.filename:
        return {"error": "Файл не передан"}, 400

    ext      = os.path.splitext(file.filename)[1].lower()
    tmp      = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_path = tmp.name
    tmp.close()
    file.save(tmp_path)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO kb_documents (original_name, folder_id, doc_type, status) "
                "VALUES (%s, %s, %s, 'pending') RETURNING id",
                (file.filename, folder_id, ext.lstrip("."))
            )
            doc_id = cur.fetchone()["id"]
        conn.commit()

    threading.Thread(
        target=_index_document_async,
        args=(str(doc_id), tmp_path, ext),
        daemon=True,
    ).start()

    return {"id": str(doc_id), "status": "pending"}, 202


@app.route("/api/kb/documents/<doc_id>", methods=["DELETE"])
def api_kb_delete_document(doc_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM kb_documents WHERE id=%s", (doc_id,))
        conn.commit()
    return {"ok": True}


@app.route("/api/kb/search", methods=["POST"])
def api_kb_search():
    data      = request.get_json(force=True)
    query     = (data.get("query") or "").strip()
    folder_id = data.get("folder_id")
    top_k     = int(data.get("top_k", 5))
    if not query:
        return {"results": []}
    results = hybrid_search(query, top_k=top_k, folder_id=folder_id)
    return {"results": results}


# ── Фоновая индексация ────────────────────────────────────────────────────────

def _index_document_async(doc_id: str, tmp_path: str, ext: str):
    try:
        if ext in (".txt", ".md", ".log"):
            text   = read_txt(tmp_path)
            chunks = list(rag_chunk_text(text))

        elif ext == ".pdf":
            if detect_pdf_type(tmp_path) == "text":
                pages  = read_pdf_text(tmp_path)
                chunks = list(rag_chunk_pages(pages))
            else:
                from pdf2image import convert_from_path
                pil_pages  = convert_from_path(tmp_path, dpi=200)
                page_texts = []
                for i, pil_page in enumerate(pil_pages):
                    b64 = to_base64(pil_page)
                    r   = _client.chat(
                        model=MODEL_DEFAULT,
                        messages=[{"role": "user",
                                   "content": f"Извлеки весь текст со страницы {i+1}.",
                                   "images": [b64]}],
                        options={"num_ctx": 4096, "temperature": 0.0},
                        stream=False,
                    )
                    page_texts.append(r["message"]["content"])
                chunks = list(rag_chunk_pages(page_texts))

        elif ext in (".docx", ".doc"):
            text   = read_docx(tmp_path)
            chunks = list(rag_chunk_text(text))

        elif ext in (".xlsx", ".xls", ".xlsm", ".ods"):
            text   = read_excel(tmp_path)
            chunks = list(rag_chunk_text(text))

        elif ext in (".csv", ".tsv"):
            text   = read_csv(tmp_path, ext)
            chunks = list(rag_chunk_text(text))

        else:
            raise ValueError(f"Неподдерживаемый формат: {ext}")

        rag_index_document(doc_id, chunks)

    except Exception as e:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE kb_documents SET status='error', meta=%s WHERE id=%s",
                    (json.dumps({"error": str(e)}), doc_id)
                )
            conn.commit()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── статика ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(ROOT_DIR, "index.html")


@app.route("/sso-login")
def sso_login():
    username = _normalize_username(request.args.get("u", ""))
    display_name = (request.args.get("d") or "").strip()
    ts_raw = (request.args.get("ts") or "").strip()
    sig = (request.args.get("sig") or "").strip()
    if not username or not display_name or not ts_raw or not sig:
        return "SSO parameters are missing", 400
    if not _verify_sso_payload(username, display_name, ts_raw, sig):
        return "SSO verification failed", 403
    session["logged_in"] = True
    session["username"] = username
    session["display_name"] = display_name
    return redirect("/")


@app.route("/api/me")
def api_me():
    username = _normalize_username(session.get("username") or "")
    display_name = (session.get("display_name") or username or "").strip()
    return jsonify({
        "logged_in": bool(session.get("logged_in")),
        "username": username,
        "display_name": display_name
    })


@app.route("/back.png")
def bg_image():
    return send_from_directory(ROOT_DIR, "back.png")

@app.route("/ico.png")
def ico_image():
    return send_from_directory(ROOT_DIR, "ico.png")

@app.route("/name.png")
def name_image():
    return send_from_directory(ROOT_DIR, "name.png")


@app.route("/api/models")
def api_models():
    available = get_available_models()
    result, seen_labels = [], set()
    for m in available:
        label = MODEL_LABELS.get(m, m)
        if label not in seen_labels:
            result.append({"id": m, "label": label})
            seen_labels.add(label)
    return {"models": result}


@app.route("/api/detect-file", methods=["POST"])
def api_detect_file():
    """Определяет тип файла и предлагает модель. Для PDF проверяет, скан это или текст."""
    file = request.files.get("file")
    if not file or not file.filename:
        return {"type": "unknown", "scanned": False, "model": MODEL_DEFAULT, "reason": ""}

    ext = os.path.splitext(file.filename)[1].lower()
    available = get_available_models()

    if ext != ".pdf":
        model, reason = suggest_model(ext, "", available)
        return {"type": ext.lstrip("."), "scanned": False, "model": model, "reason": reason}

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()
    file.save(tmp_path)
    try:
        is_scanned = detect_pdf_type(tmp_path) == "scanned"
        eff_ext = ".jpg" if is_scanned else ".pdf"
        model, reason = suggest_model(eff_ext, "", available)
        if is_scanned:
            reason = "Сканированный PDF — нужна vision-модель"
        return {"type": "pdf", "scanned": is_scanned, "model": model, "reason": reason}
    finally:
        os.unlink(tmp_path)


@app.route("/api/suggest-model", methods=["POST"])
def api_suggest_model():
    data = request.get_json(force=True)
    file_ext = data.get("file_ext", "").lower()
    question = data.get("question", "")
    available = get_available_models()
    model, reason = suggest_model(file_ext, question, available)
    return {"model": model, "reason": reason}


if __name__ == "__main__":
    init_db()
    print("🚀 Запуск на http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
