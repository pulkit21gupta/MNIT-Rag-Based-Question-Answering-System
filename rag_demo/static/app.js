/* ═══════════════════════════════════════════════
   MNIT RAG Assistant — Frontend JS
   ═══════════════════════════════════════════════ */

const API = '';  // same origin

// ── DOM refs ─────────────────────────────────────
const chatArea     = document.getElementById('chatArea');
const messages     = document.getElementById('messages');
const welcome      = document.getElementById('welcome');
const questionInput= document.getElementById('questionInput');
const sendBtn      = document.getElementById('sendBtn');
const clearBtn     = document.getElementById('clearBtn');
const menuBtn      = document.getElementById('menuBtn');
const sidebar      = document.getElementById('sidebar');
const sourceList   = document.getElementById('sourceList');
const statusDot    = document.getElementById('statusDot');
const statusText   = document.getElementById('statusText');

// ── State ─────────────────────────────────────────
let systemReady = false;

// ── Marked config ─────────────────────────────────
marked.setOptions({ breaks: true, gfm: true });

// ── Sidebar toggle ────────────────────────────────
menuBtn.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
});

// ── Auto-resize textarea ──────────────────────────
questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + 'px';
  sendBtn.disabled = !questionInput.value.trim() || !systemReady;
});

// ── Send on Enter (Shift+Enter = newline) ─────────
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) handleSend();
  }
});

sendBtn.addEventListener('click', handleSend);

// ── Suggestion chips ──────────────────────────────
document.querySelectorAll('.suggestion-chip').forEach(btn => {
  btn.addEventListener('click', () => {
    questionInput.value = btn.dataset.q;
    questionInput.dispatchEvent(new Event('input'));
    handleSend();
  });
});

// ── Clear conversation ────────────────────────────
clearBtn.addEventListener('click', () => {
  messages.innerHTML = '';
  welcome.style.display = 'flex';
  welcome.style.minHeight = '100%';
});

// ── Status polling ────────────────────────────────
function setStatus(ready, msg) {
  statusText.textContent = msg;
  statusDot.className    = 'status-dot ' + (ready ? 'ready' : 'busy');
  systemReady            = ready;
  sendBtn.disabled       = !ready || !questionInput.value.trim();
}

async function pollStatus() {
  try {
    const res  = await fetch(`${API}/api/status`);
    const data = await res.json();
    setStatus(data.ready, data.message);
    if (!data.ready) setTimeout(pollStatus, 2000);
    else             loadSources();
  } catch {
    setStatus(false, 'Connecting…');
    setTimeout(pollStatus, 3000);
  }
}

// ── Load knowledge-base sources ───────────────────
async function loadSources() {
  try {
    const res   = await fetch(`${API}/api/sources`);
    const data  = await res.json();
    const files = data.sources || [];

    sourceList.innerHTML = '';
    if (!files.length) {
      sourceList.innerHTML = '<li class="source-item" style="color:var(--text-dim)">No sources found</li>';
      return;
    }
    files.forEach(f => {
      const li   = document.createElement('li');
      li.className = 'source-item';
      const icon   = getFileIcon(f);
      li.innerHTML = `<span class="file-icon">${icon}</span><span>${f}</span>`;
      sourceList.appendChild(li);
    });
  } catch {
    sourceList.innerHTML = '<li class="source-item" style="color:var(--text-dim)">Failed to load</li>';
  }
}

function getFileIcon(filename) {
  if (filename.toLowerCase().includes('hostel'))      return '🏠';
  if (filename.toLowerCase().includes('placement'))   return '💼';
  if (filename.toLowerCase().includes('academic'))    return '📚';
  if (filename.toLowerCase().includes('sport'))       return '⚽';
  if (filename.toLowerCase().includes('bank'))        return '🏦';
  if (filename.toLowerCase().includes('cultural'))    return '🎭';
  if (filename.toLowerCase().includes('admin'))       return '🏛️';
  if (filename.toLowerCase().includes('guest'))       return '🏨';
  if (filename.toLowerCase().includes('post'))        return '📮';
  if (filename.toLowerCase().includes('alumni'))      return '🎓';
  if (filename.toLowerCase().includes('aide') || filename.toLowerCase().includes('department')) return '🔬';
  return '📄';
}

// ── Message helpers ───────────────────────────────
function hideWelcome() {
  welcome.style.display = 'none';
  welcome.style.minHeight = '0';
}

function appendMessage(role, html, sources) {
  const isAI   = role === 'ai';
  const msg    = document.createElement('div');
  msg.className = `msg ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${role}`;
  avatar.textContent = isAI ? 'M' : 'U';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  const content = document.createElement('div');
  content.className = 'bubble-content';
  if (isAI) {
    content.innerHTML = html;
  } else {
    content.textContent = html;
  }
  bubble.appendChild(content);

  if (isAI && sources && sources.length) {
    const chips = document.createElement('div');
    chips.className = 'source-chips';
    sources.forEach(s => {
      const chip = document.createElement('span');
      chip.className = 'source-chip';
      chip.innerHTML = `<span class="source-chip-icon">${getFileIcon(s.file)}</span>${s.file}`;
      chip.title = s.snippet;
      chips.appendChild(chip);
    });
    bubble.appendChild(chips);
  }

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  messages.appendChild(msg);
  scrollToBottom();
  return msg;
}

function appendTypingIndicator() {
  const msg    = document.createElement('div');
  msg.className = 'msg ai typing-indicator';
  msg.id = 'typing';

  const avatar = document.createElement('div');
  avatar.className = 'avatar ai';
  avatar.textContent = 'M';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  const content = document.createElement('div');
  content.className = 'bubble-content';
  content.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

  bubble.appendChild(content);
  msg.appendChild(avatar);
  msg.appendChild(bubble);
  messages.appendChild(msg);
  scrollToBottom();
  return msg;
}

function removeTypingIndicator() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

function scrollToBottom() {
  chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
}

// ── Main send handler ─────────────────────────────
async function handleSend() {
  const question = questionInput.value.trim();
  if (!question || !systemReady) return;

  hideWelcome();

  // Reset input
  questionInput.value = '';
  questionInput.style.height = 'auto';
  sendBtn.disabled = true;

  // User message
  appendMessage('user', question, null);

  // Typing indicator
  appendTypingIndicator();

  // Disable input during request
  questionInput.disabled = true;
  setStatus(false, 'Thinking…');
  statusDot.className = 'status-dot busy';

  try {
    const res  = await fetch(`${API}/api/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question }),
    });

    const data = await res.json();
    removeTypingIndicator();

    if (data.error) {
      const errMsg = document.createElement('div');
      errMsg.className = 'msg ai error-bubble';
      errMsg.innerHTML = `
        <div class="avatar ai">M</div>
        <div class="bubble">
          <div class="bubble-content">⚠️ ${data.error}</div>
        </div>`;
      messages.appendChild(errMsg);
    } else {
      const html = marked.parse(data.answer || '');
      appendMessage('ai', html, data.sources || []);
    }
  } catch (err) {
    removeTypingIndicator();
    const errMsg = document.createElement('div');
    errMsg.className = 'msg ai error-bubble';
    errMsg.innerHTML = `
      <div class="avatar ai">M</div>
      <div class="bubble">
        <div class="bubble-content">⚠️ Connection error: ${err.message}</div>
      </div>`;
    messages.appendChild(errMsg);
  } finally {
    questionInput.disabled = false;
    questionInput.focus();
    setStatus(true, 'Ready');
    scrollToBottom();
  }
}

// ── Boot ──────────────────────────────────────────
pollStatus();
