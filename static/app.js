let currentFile = null, isStreaming = false;
let abortController = null;
let currentConvId = null;
let autoModelEnabled = true;
let userOverrodeModel = false;

const sendIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`;
const stopIcon  = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>`;
const cpIco = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>`;
const okIco  = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;

// ── copy ─────────────────────────────────────────────────────────────────────

function htmlToReadable(html) {
  const d = document.createElement('div');
  d.innerHTML = html;
  return nodeToText(d).replace(/\n{3,}/g, '\n\n').trim();
}

function nodeToText(n) {
  if (n.nodeType === Node.TEXT_NODE) return n.textContent;
  const t = n.tagName ? n.tagName.toLowerCase() : '';
  const ch = () => Array.from(n.childNodes).map(nodeToText).join('');
  if (t === 'pre') {
    const c = n.querySelector('code');
    const tx = (c || n).innerText;
    const lg = (c && c.className.match(/language-(\w+)/)?.[1]) || '';
    return '\n```' + lg + '\n' + tx.trimEnd() + '\n```\n';
  }
  if (t === 'code') return '`' + n.textContent + '`';
  if (t === 'table') return tblAscii(n) + '\n';
  if (['thead','tbody','tfoot','tr','th','td'].includes(t)) return ch();
  if (t === 'h1') return '\n# '   + ch().trim() + '\n';
  if (t === 'h2') return '\n## '  + ch().trim() + '\n';
  if (t === 'h3') return '\n### ' + ch().trim() + '\n';
  if (t === 'ul') return '\n' + Array.from(n.children).map(li => '• ' + nodeToText(li).trim()).join('\n') + '\n';
  if (t === 'ol') return '\n' + Array.from(n.children).map((li, i) => (i+1) + '. ' + nodeToText(li).trim()).join('\n') + '\n';
  if (t === 'li') return ch();
  if (['p','div','blockquote','section','article'].includes(t)) return '\n' + ch().trim() + '\n';
  if (t === 'br') return '\n';
  if (t === 'hr') return '\n' + '─'.repeat(36) + '\n';
  return ch();
}

function tblAscii(el) {
  const rows = [];
  el.querySelectorAll('tr').forEach(r =>
    rows.push(Array.from(r.querySelectorAll('th,td')).map(c => c.innerText.replace(/\n/g,' ').trim()))
  );
  if (!rows.length) return '';
  const cols = Math.max(...rows.map(r => r.length));
  rows.forEach(r => { while (r.length < cols) r.push(''); });
  const ws = Array.from({length: cols}, (_, c) => Math.max(...rows.map(r => r[c].length), 3));
  const sep = '+-' + ws.map(w => '-'.repeat(w)).join('-+-') + '-+';
  const row = cs => '| ' + cs.map((c,i) => c.padEnd(ws[i])).join(' | ') + ' |';
  const ls = [sep];
  rows.forEach((r,i) => { ls.push(row(r)); if (i===0) ls.push(sep); });
  ls.push(sep);
  return '\n' + ls.join('\n') + '\n';
}

function tblPlain(el) {
  const rows = [];
  el.querySelectorAll('tr').forEach(r =>
    rows.push(Array.from(r.querySelectorAll('th,td')).map(c => c.innerText.replace(/\n+/g,' ').trim()).join('\t'))
  );
  return rows.join('\n').trim();
}

function doCopy(text, btn) {
  const iconOnly = btn.classList.contains('tcbtn');
  const ok = () => {
    btn.classList.add('ok');
    btn.innerHTML = iconOnly ? okIco : okIco + ' copied';
    setTimeout(() => {
      btn.classList.remove('ok');
      btn.innerHTML = iconOnly ? cpIco : cpIco + (btn.dataset.lbl || ' copy');
    }, 2000);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(ok).catch(() => fbCopy(text, ok));
  } else {
    fbCopy(text, ok);
  }
}

function fbCopy(text, cb) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.cssText = 'position:fixed;top:-9999px;opacity:0';
  document.body.appendChild(ta);
  ta.focus(); ta.select();
  try { document.execCommand('copy'); cb(); } catch(e) {}
  document.body.removeChild(ta);
}

function addCodeBtns(el) {
  el.querySelectorAll('pre').forEach(pre => {
    if (pre.parentElement.classList.contains('code-wrap')) return;
    const w = document.createElement('div');
    w.className = 'code-wrap';
    pre.parentNode.insertBefore(w, pre);
    w.appendChild(pre);
    const b = document.createElement('button');
    b.className = 'cbtn'; b.dataset.lbl = ' copy';
    b.innerHTML = cpIco + ' copy';
    b.onclick = () => doCopy(pre.innerText.trimEnd(), b);
    w.appendChild(b);
  });

  el.querySelectorAll('table').forEach(table => {
    if (table.parentElement.classList.contains('table-wrap')) return;
    const outer = document.createElement('div');
    outer.className = 'table-outer';
    table.parentNode.insertBefore(outer, table);
    const tools = document.createElement('div');
    tools.className = 'table-tools';
    outer.appendChild(tools);
    const w = document.createElement('div');
    w.className = 'table-wrap';
    outer.appendChild(w);
    w.appendChild(table);
    const b = document.createElement('button');
    b.className = 'tcbtn';
    b.title = 'Copy table';
    b.setAttribute('aria-label', 'Copy table');
    b.innerHTML = cpIco;
    b.onclick = () => doCopy(tblPlain(table), b);
    tools.appendChild(b);
  });
}

function addMsgCopy(el, html) {
  const e = el.querySelector('.mcopy'); if (e) e.remove();
  const b = document.createElement('button');
  b.className = 'mcopy'; b.dataset.lbl = ' копировать ответ';
  b.innerHTML = cpIco + ' копировать ответ';
  b.onclick = () => doCopy(htmlToReadable(html), b);
  el.appendChild(b);
}

// ── markdown ─────────────────────────────────────────────────────────────────

function fixMarkdown(md) {
  return md.replace(/^```[^\n]*\n([\s\S]*?)^```/gm, (match, inner) => {
    const lines = inner.trim().split('\n');
    const isTable = lines.length >= 2
      && lines[0].trim().startsWith('|')
      && /^\|[\s|:\-]+\|$/.test(lines[1].trim());
    return isTable ? inner.trim() : match;
  });
}

// ── conversations ─────────────────────────────────────────────────────────────

async function loadConversations() {
  try {
    const data = await fetch('/api/conversations').then(r => r.json());
    renderConvList(data.conversations || []);
  } catch(e) {
    console.error(e);
  }
}

function renderConvList(convs) {
  const list = document.getElementById('chats-list');
  if (!convs.length) {
    list.innerHTML = '<div class="chats-empty">Нет чатов</div>';
    return;
  }
  list.innerHTML = convs.map(c => `
    <div class="chat-item ${c.id === currentConvId ? 'active' : ''}" data-id="${c.id}" onclick="openChat('${c.id}')">
      <div class="chat-item-title">${esc(c.title)}</div>
      <button class="chat-item-del" onclick="deleteChat(event,'${c.id}')" title="Удалить">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
  `).join('');
}

async function newChat() {
  const model = document.getElementById('model-select').value || 'qwen2.5:3b';
  const data = await fetch('/api/conversations', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model})
  }).then(r => r.json());

  currentConvId = data.id;
  clearFeed();
  await loadConversations();
  highlightActiveChat();
}

async function openChat(id) {
  if (isStreaming) return;
  currentConvId = id;
  clearFeed();
  highlightActiveChat();

  const data = await fetch(`/api/conversations/${id}/messages`).then(r => r.json());
  const msgs = data.messages || [];
  if (!msgs.length) return;

  hideEmpty();
  for (const m of msgs) {
    const ac = addMessage(m.role, m.role === 'user' ? m.content : '', m.file_name || null);
    if (m.role === 'assistant') {
      ac.innerHTML = marked.parse(fixMarkdown(m.content));
      addCodeBtns(ac);
      addMsgCopy(ac, ac.innerHTML);
    }
  }
  scrollEnd();
}

async function deleteChat(e, id) {
  e.stopPropagation();
  await fetch(`/api/conversations/${id}`, {method: 'DELETE'});
  if (currentConvId === id) {
    currentConvId = null;
    clearFeed();
    document.getElementById('empty-state').style.display = '';
  }
  loadConversations();
}

function highlightActiveChat() {
  document.querySelectorAll('.chat-item').forEach(el => {
    el.classList.toggle('active', el.dataset.id === currentConvId);
  });
}

function clearFeed() {
  const inner = document.getElementById('chat-inner');
  inner.innerHTML = '';
  const welcome = document.createElement('div');
  welcome.className = 'welcome';
  welcome.id = 'empty-state';
  welcome.innerHTML = `
    <div class="welcome-title">Добрый день</div>
    <div class="welcome-sub">Загрузите документ или задайте вопрос — анализирую PDF, DOCX, Excel, изображения и другие форматы</div>
    <div class="chips">
      <div class="chip" onclick="setPrompt('Кратко изложи содержание')">Кратко изложи содержание</div>
      <div class="chip" onclick="setPrompt('Выдели ключевые факты и цифры')">Ключевые факты и цифры</div>
      <div class="chip" onclick="setPrompt('Какие обязательства сторон?')">Обязательства сторон</div>
      <div class="chip" onclick="setPrompt('Сделай структурированное резюме')">Структурированное резюме</div>
    </div>`;
  inner.appendChild(welcome);
}

// ── models ────────────────────────────────────────────────────────────────────

async function loadModels() {
  try {
    const d = await fetch('/api/models').then(r => r.json());
    const sel = document.getElementById('model-select');
    sel.innerHTML = '';
    (d.models && d.models.length ? d.models : ['qwen2.5:3b']).forEach(m => {
      const o = document.createElement('option');
      o.value = m; o.textContent = m;
      sel.appendChild(o);
    });
  } catch {
    document.getElementById('stat-status').textContent = 'Нет соединения';
    document.getElementById('status-dot').classList.add('off');
  }
}

// ── textarea ──────────────────────────────────────────────────────────────────

const ta = document.getElementById('input-field');
ta.addEventListener('input', () => {
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
});
ta.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ── file handling ─────────────────────────────────────────────────────────────

document.getElementById('file-input').addEventListener('change', e => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

document.addEventListener('paste', e => {
  const items = (e.clipboardData || e.originalEvent?.clipboardData)?.items;
  if (!items) return;
  for (const item of items) {
    if (item.kind === 'file') {
      const f = item.getAsFile();
      if (f) { setFile(f); showToast('Файл вставлен: ' + f.name); }
      break;
    }
  }
});

function setFile(f) {
  currentFile = f;
  document.getElementById('file-preview-name').textContent = f.name;
  document.getElementById('file-preview-size').textContent = fmtSize(f.size);
  document.getElementById('file-preview').classList.add('show');
  if (autoModelEnabled && !userOverrodeModel) {
    detectFileAndSelectModel(f);
  }
}

async function detectFileAndSelectModel(f) {
  const ext = '.' + f.name.split('.').pop().toLowerCase();
  if (ext !== '.pdf') {
    await autoSelectModel(ext, document.getElementById('input-field').value.trim());
    return;
  }
  // PDF: отправляем на сервер — он проверит скан это или текст
  try {
    const fd = new FormData();
    fd.append('file', f);
    const data = await fetch('/api/detect-file', { method: 'POST', body: fd }).then(r => r.json());
    const sel = document.getElementById('model-select');
    if (data.model && sel.querySelector(`option[value="${data.model}"]`)) {
      sel.value = data.model;
      showAutoModelBadge(data.model, data.reason);
    }
  } catch(e) {
    await autoSelectModel(ext, document.getElementById('input-field').value.trim());
  }
}

function removeFile() {
  currentFile = null;
  document.getElementById('file-input').value = '';
  document.getElementById('file-preview').classList.remove('show');
  clearAutoModelBadge();
}

function fmtSize(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(1) + ' MB';
}

// ── drag & drop ───────────────────────────────────────────────────────────────

let dragN = 0;
document.addEventListener('dragenter', e => { e.preventDefault(); dragN++; document.getElementById('drop-overlay').classList.add('show'); });
document.addEventListener('dragleave', e => { dragN--; if (dragN <= 0) { dragN = 0; document.getElementById('drop-overlay').classList.remove('show'); } });
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault(); dragN = 0;
  document.getElementById('drop-overlay').classList.remove('show');
  const f = e.dataTransfer.files[0];
  if (f) { setFile(f); showToast('Файл прикреплён: ' + f.name); }
});

// ── messages ──────────────────────────────────────────────────────────────────

function hideEmpty() {
  const el = document.getElementById('empty-state');
  if (el) el.style.display = 'none';
}

function addMessage(role, content, badge) {
  hideEmpty();
  const wrap = document.getElementById('chat-inner');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const av = role === 'user'
    ? `<div class="msg-avatar">↑</div>`
    : `<div class="msg-avatar"><img src="/ico.png" alt="" style="width:20px;height:20px;object-fit:contain;border-radius:6px;"></div>`;
  const bd = badge ? `<div class="msg-file-badge">${esc(badge)}</div><br>` : '';
  div.innerHTML = `${av}<div class="msg-body"><div class="msg-role">${role === 'user' ? 'Вы' : 'БелнипиAI'}</div><div class="msg-content">${bd}${role === 'user' ? esc(content) : ''}</div></div>`;
  wrap.appendChild(div);
  return div.querySelector('.msg-content');
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function scrollEnd() { const a = document.getElementById('chat-area'); a.scrollTop = a.scrollHeight; }

// ── stop ──────────────────────────────────────────────────────────────────────

function stopStreaming() {
  if (abortController) { abortController.abort(); abortController = null; }
}

// ── send ──────────────────────────────────────────────────────────────────────

async function sendMessage() {
  if (isStreaming) return;
  const q = ta.value.trim(); if (!q) return;

  // Авто-выбор для вопроса без файла (сложные запросы)
  if (autoModelEnabled && !userOverrodeModel && !currentFile) {
    await autoSelectModel('', q);
  }

  const model = document.getElementById('model-select').value || 'qwen2.5:3b';
  userOverrodeModel = false;
  clearAutoModelBadge();

  // Создаём чат если нет
  if (!currentConvId) {
    const data = await fetch('/api/conversations', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model})
    }).then(r => r.json());
    currentConvId = data.id;
    await loadConversations();
    highlightActiveChat();
  }

  isStreaming = true;
  abortController = new AbortController();
  const btn = document.getElementById('send-btn');
  btn.classList.add('stop');
  btn.innerHTML = stopIcon;
  btn.onclick = stopStreaming;
  ta.value = ''; ta.style.height = 'auto';

  const fc = currentFile, fn = fc ? fc.name : null;
  if (fc) removeFile();

  addMessage('user', q, fn); scrollEnd();
  const ac = addMessage('assistant', '');
  const dots = document.createElement('div');
  dots.className = 'thinking';
  dots.innerHTML = '<span></span><span></span><span></span>';
  ac.appendChild(dots); scrollEnd();

  const fd = new FormData();
  fd.append('question', q); fd.append('model', model);
  if (fc) fd.append('file', fc);

  let raw = '', first = true;
  try {
    const res = await fetch(`/api/conversations/${currentConvId}/chat`, {
      method: 'POST', body: fd, signal: abortController.signal
    });
    const reader = res.body.getReader(), dec = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      for (const line of dec.decode(value).split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const p = line.slice(6); if (p === '[DONE]') break;
        try {
          const o = JSON.parse(p);
          if (o.error) { showToast(o.error, true); break; }
          if (o.title) {
            updateChatTitle(currentConvId, o.title);
          }
          if (o.text) {
            if (first) { dots.remove(); first = false; }
            raw += o.text;
            ac.innerHTML = marked.parse(fixMarkdown(raw)) + '<span class="cursor"></span>';
            addCodeBtns(ac); scrollEnd();
          }
        } catch(e) {}
      }
    }
  } catch(err) {
    if (first) dots.remove();
    if (err.name !== 'AbortError') {
      ac.innerHTML = '<span style="color:var(--red)">Ошибка соединения с сервером</span>';
      showToast('Ошибка соединения', true);
    }
  }

  const cur = ac.querySelector('.cursor'); if (cur) cur.remove();
  if (first) dots.remove();
  if (raw) {
    ac.innerHTML = marked.parse(fixMarkdown(raw));
    addCodeBtns(ac);
    addMsgCopy(ac, ac.innerHTML);
  }

  abortController = null;
  isStreaming = false;
  btn.classList.remove('stop');
  btn.innerHTML = sendIcon;
  btn.onclick = sendMessage;
  ta.focus(); scrollEnd();
}

function updateChatTitle(id, title) {
  const item = document.querySelector(`.chat-item[data-id="${id}"] .chat-item-title`);
  if (item) item.textContent = title;
}

// ── auto model ────────────────────────────────────────────────────────────────

async function autoSelectModel(fileExt, question) {
  if (!autoModelEnabled) return;
  try {
    const data = await fetch('/api/suggest-model', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({file_ext: fileExt, question})
    }).then(r => r.json());

    const sel = document.getElementById('model-select');
    if (data.model && sel.querySelector(`option[value="${data.model}"]`)) {
      sel.value = data.model;
      showAutoModelBadge(data.model, data.reason);
    }
  } catch(e) {}
}

function showAutoModelBadge(model, reason) {
  let badge = document.getElementById('auto-model-badge');
  if (!badge) {
    badge = document.createElement('span');
    badge.id = 'auto-model-badge';
    document.querySelector('.field-label').appendChild(badge);
  }
  badge.className = 'auto-model-badge';
  badge.title = reason;
  badge.innerHTML = `<span class="auto-model-label">авто</span><span class="auto-model-name">${esc(model)}</span>`;
  badge.style.display = 'inline-flex';
}

function clearAutoModelBadge() {
  const badge = document.getElementById('auto-model-badge');
  if (badge) badge.style.display = 'none';
}

// ── helpers ───────────────────────────────────────────────────────────────────

function setPrompt(t) { ta.value = t; ta.focus(); ta.dispatchEvent(new Event('input')); }

function showToast(msg, err) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show' + (err ? ' error' : '');
  setTimeout(() => t.classList.remove('show'), 3000);
}

// ── nav collapse ──────────────────────────────────────────────────────────────

function toggleNav() {
  const shell = document.querySelector('.shell');
  const nav = document.querySelector('.nav');
  const isNowCollapsed = shell.classList.toggle('nav-collapsed');
  nav.classList.toggle('collapsed', isNowCollapsed);
  localStorage.setItem('navCollapsed', isNowCollapsed ? 'true' : 'false');
}

function initNavState() {
  const saved = localStorage.getItem('navCollapsed');
  const isSmall = window.innerWidth < 900;
  const shouldCollapse = saved === 'true' || (saved === null && isSmall);
  if (shouldCollapse) {
    document.querySelector('.shell').classList.add('nav-collapsed');
    document.querySelector('.nav').classList.add('collapsed');
  }
}

// ── delete all chats ──────────────────────────────────────────────────────────

async function deleteAllChats() {
  const items = document.querySelectorAll('.chat-item');
  if (!items.length) return;
  if (!confirm('Удалить все чаты? Это действие нельзя отменить.')) return;
  const ids = Array.from(items).map(el => el.dataset.id);
  await Promise.all(ids.map(id => fetch(`/api/conversations/${id}`, { method: 'DELETE' })));
  currentConvId = null;
  clearFeed();
  loadConversations();
  showToast('Все чаты удалены');
}

// ── init ──────────────────────────────────────────────────────────────────────

marked.use({ gfm: true });
initNavState();
loadModels();
loadConversations();
ta.focus();

// Сбрасываем авто при ручном выборе
document.getElementById('model-select').addEventListener('change', () => {
  userOverrodeModel = true;
  clearAutoModelBadge();
});
