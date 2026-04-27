/* ============================================================
   EDU Agent — Frontend Application Logic
   ============================================================ */

const API = '';  // same origin

// ============ STATE ============
let currentSubject = null;
let quizScore = 0;
let quizTotal = 0;
let selectedFiles = [];

// ============ DOM REFS ============
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
    sidebar: $('#sidebar'),
    sidebarToggle: $('#sidebarToggle'),
    sidebarClose: $('#sidebarClose'),
    subjectSelect: $('#subjectSelect'),
    newSubjectBtn: $('#newSubjectBtn'),
    newSubjectForm: $('#newSubjectForm'),
    newSubjectName: $('#newSubjectName'),
    newSubjectIcon: $('#newSubjectIcon'),
    createSubjectBtn: $('#createSubjectBtn'),
    cancelSubjectBtn: $('#cancelSubjectBtn'),
    currentSubjectIcon: $('#currentSubjectIcon'),
    currentSubjectName: $('#currentSubjectName'),
    connectionStatus: $('#connectionStatus'),
    statsSection: $('#statsSection'),
    statVectors: $('#statVectors'),
    // Chat
    chatContainer: $('#chatContainer'),
    welcomeScreen: $('#welcomeScreen'),
    messages: $('#messages'),
    chatInputArea: $('#chatInputArea'),
    chatInput: $('#chatInput'),
    sendBtn: $('#sendBtn'),
    // Quiz
    lectureSelect: $('#lectureSelect'),
    generateQuizBtn: $('#generateQuizBtn'),
    shuffleQuizBtn: $('#shuffleQuizBtn'),
    quizScore: $('#quizScore'),
    quizQuestions: $('#quizQuestions'),
    scoreValue: $('#scoreValue'),
    scoreTotal: $('#scoreTotal'),
    scoreFill: $('#scoreFill'),
    // Upload
    uploadZone: $('#uploadZone'),
    fileInput: $('#fileInput'),
    fileList: $('#fileList'),
    processBtn: $('#processBtn'),
    processLog: $('#processLog'),
    logEntry: $('#logEntry'),
    progressFill: $('#progressFill'),
    // Loading
    loadingOverlay: $('#loadingOverlay'),
    loadingText: $('#loadingText'),
};

// ============ INIT ============
document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    initTabs();
    initChat();
    initUpload();
    initQuiz();
    loadSubjects();
});

// ============ SIDEBAR ============
function initSidebar() {
    els.sidebarToggle.addEventListener('click', () => els.sidebar.classList.toggle('open'));
    els.sidebarClose.addEventListener('click', () => els.sidebar.classList.remove('open'));

    els.newSubjectBtn.addEventListener('click', () => {
        els.newSubjectForm.classList.remove('hidden');
        els.newSubjectName.focus();
    });
    els.cancelSubjectBtn.addEventListener('click', () => els.newSubjectForm.classList.add('hidden'));
    els.createSubjectBtn.addEventListener('click', createSubject);

    els.subjectSelect.addEventListener('change', () => {
        const key = els.subjectSelect.value;
        if (key) loadSubject(key);
    });
}

// ============ TABS ============
function initTabs() {
    $$('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            $$('.nav-item').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            $$('.tab-content').forEach(t => t.classList.remove('active'));
            $(`#tab${tab.charAt(0).toUpperCase() + tab.slice(1)}`).classList.add('active');
            els.sidebar.classList.remove('open');
        });
    });
}

// ============ SUBJECTS ============
async function loadSubjects() {
    try {
        const resp = await fetch(`${API}/api/subjects`);
        const subjects = await resp.json();
        els.subjectSelect.innerHTML = '<option value="">— Select Subject —</option>';
        for (const [key, subj] of Object.entries(subjects)) {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = `${subj.icon} ${subj.name}`;
            els.subjectSelect.appendChild(opt);
        }
        // Auto-load first
        const keys = Object.keys(subjects);
        if (keys.length > 0) {
            els.subjectSelect.value = keys[0];
            loadSubject(keys[0]);
        }
    } catch (e) {
        console.error('Failed to load subjects:', e);
    }
}

async function loadSubject(key) {
    showLoading('Loading knowledge base...');
    setStatus('loading', 'Loading...');
    try {
        const resp = await fetch(`${API}/api/subjects/${key}/load`, { method: 'POST' });
        const data = await resp.json();
        if (data.loaded) {
            currentSubject = key;
            const subjects = await (await fetch(`${API}/api/subjects`)).json();
            const subj = subjects[key];
            els.currentSubjectIcon.textContent = subj.icon;
            els.currentSubjectName.textContent = subj.name;
            els.statVectors.textContent = data.vectors;
            els.statsSection.style.display = 'block';
            els.welcomeScreen.style.display = 'none';
            els.chatInputArea.style.display = 'block';
            setStatus('online', `${data.vectors} vectors`);
            loadLectures();
        } else {
            setStatus('offline', data.message || 'Not loaded');
            els.welcomeScreen.style.display = 'flex';
            els.chatInputArea.style.display = 'none';
        }
    } catch (e) {
        setStatus('offline', 'Error');
        console.error(e);
    }
    hideLoading();
}

async function createSubject() {
    const name = els.newSubjectName.value.trim();
    const icon = els.newSubjectIcon.value;
    if (!name) return;
    try {
        await fetch(`${API}/api/subjects`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, icon }),
        });
        els.newSubjectForm.classList.add('hidden');
        els.newSubjectName.value = '';
        await loadSubjects();
    } catch (e) {
        alert('Error creating subject: ' + e.message);
    }
}

// ============ CHAT ============
function initChat() {
    els.sendBtn.addEventListener('click', sendMessage);
    els.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

async function sendMessage() {
    const q = els.chatInput.value.trim();
    if (!q || !currentSubject) return;

    els.chatInput.value = '';
    els.sendBtn.disabled = true;

    addMessage('user', q);
    const typingEl = addTypingIndicator();

    try {
        const resp = await fetch(`${API}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q }),
        });
        const data = await resp.json();
        typingEl.remove();

        if (resp.ok) {
            addMessage('assistant', data.answer, data.sources, data.confidence);
        } else {
            addMessage('assistant', `❌ Error: ${data.detail || 'Unknown error'}`);
        }
    } catch (e) {
        typingEl.remove();
        addMessage('assistant', `❌ Connection error: ${e.message}`);
    }

    els.sendBtn.disabled = false;
    els.chatInput.focus();
}

function addMessage(role, text, sources = [], confidence = null) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const avatar = role === 'user' ? '👤' : '🎓';
    const formatted = formatMarkdown(text);

    let html = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-body">${formatted}`;

    if (sources && sources.length > 0) {
        html += `<div class="message-sources">📚 Sources: ${sources.map(s => `<br>• ${s}`).join('')}</div>`;
    }
    if (confidence) {
        const icons = { high: '🟢', medium: '🟡', low: '🔴' };
        html += `<div class="confidence-badge ${confidence}">${icons[confidence] || '🟡'} ${confidence}</div>`;
    }

    html += '</div>';
    div.innerHTML = html;
    els.messages.appendChild(div);
    els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
    return div;
}

function addTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="message-avatar">🎓</div>
        <div class="message-body typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>`;
    els.messages.appendChild(div);
    els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
    return div;
}

function formatMarkdown(text) {
    if (!text) return '';
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>')
        .replace(/^[•\-]\s*(.+)$/gm, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
        .replace(/\n{2,}/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

// ============ QUIZ ============
function initQuiz() {
    els.generateQuizBtn.addEventListener('click', () => generateQuiz(false));
    els.shuffleQuizBtn.addEventListener('click', () => generateQuiz(false));
}

async function loadLectures() {
    try {
        const resp = await fetch(`${API}/api/lectures`);
        const lectures = await resp.json();
        els.lectureSelect.innerHTML = '<option value="">— Select Lecture —</option>';
        lectures.forEach(lec => {
            const opt = document.createElement('option');
            opt.value = lec;
            opt.textContent = lec.replace(/\+/g, ' ').replace(/_/g, ' ');
            els.lectureSelect.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load lectures:', e);
    }
}

async function generateQuiz(regenerate = false) {
    const lecture = els.lectureSelect.value;
    if (!lecture) return alert('Please select a lecture first!');

    els.generateQuizBtn.disabled = true;
    showLoading('Generating academic MCQ questions... This may take 30-60 seconds.');

    try {
        const resp = await fetch(`${API}/api/quiz/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lecture, regenerate }),
        });
        const data = await resp.json();

        if (resp.ok && data.questions) {
            renderQuiz(data.questions);
            els.shuffleQuizBtn.style.display = 'inline-flex';
        } else {
            alert('Error: ' + (data.detail || 'Failed to generate quiz'));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }

    els.generateQuizBtn.disabled = false;
    hideLoading();
}

function renderQuiz(questions) {
    quizScore = 0;
    quizTotal = questions.length;
    els.scoreValue.textContent = '0';
    els.scoreTotal.textContent = quizTotal;
    els.scoreFill.style.width = '0%';
    els.quizScore.style.display = 'block';
    els.quizQuestions.innerHTML = '';

    questions.forEach((q, i) => {
        if (q.raw) {
            // Fallback: raw text
            const card = document.createElement('div');
            card.className = 'question-card';
            card.innerHTML = `<div class="question-text">${formatMarkdown(q.raw)}</div>`;
            els.quizQuestions.appendChild(card);
            return;
        }

        const card = document.createElement('div');
        card.className = 'question-card';
        card.style.animationDelay = `${i * 0.1}s`;

        let choicesHtml = '';
        for (const [letter, text] of Object.entries(q.choices || {})) {
            choicesHtml += `
                <button class="choice-btn" data-letter="${letter}" data-correct="${q.correct}" data-qid="${q.id}">
                    <span class="choice-letter">${letter}</span>
                    <span>${text}</span>
                </button>`;
        }

        card.innerHTML = `
            <div class="question-number">Question ${q.id}</div>
            <div class="question-text">${q.question}</div>
            <div class="choices">${choicesHtml}</div>`;

        els.quizQuestions.appendChild(card);
    });

    // Attach click handlers
    $$('.choice-btn').forEach(btn => {
        btn.addEventListener('click', handleChoice);
    });
}

function handleChoice(e) {
    const btn = e.currentTarget;
    const letter = btn.dataset.letter;
    const correct = btn.dataset.correct;
    const qid = btn.dataset.qid;

    // Mark all choices in this question as answered
    const siblings = btn.parentElement.querySelectorAll('.choice-btn');
    siblings.forEach(s => s.classList.add('answered'));

    if (letter === correct) {
        btn.classList.add('correct');
        quizScore++;
    } else {
        btn.classList.add('wrong');
        // Highlight the correct one
        siblings.forEach(s => {
            if (s.dataset.letter === correct) s.classList.add('correct');
        });
    }

    els.scoreValue.textContent = quizScore;
    els.scoreFill.style.width = `${(quizScore / quizTotal) * 100}%`;
}

// ============ UPLOAD ============
function initUpload() {
    els.uploadZone.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', handleFiles);

    els.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        els.uploadZone.classList.add('dragover');
    });
    els.uploadZone.addEventListener('dragleave', () => els.uploadZone.classList.remove('dragover'));
    els.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        els.uploadZone.classList.remove('dragover');
        handleDroppedFiles(e.dataTransfer.files);
    });

    els.processBtn.addEventListener('click', processLectures);
}

function handleFiles(e) {
    handleDroppedFiles(e.target.files);
}

function handleDroppedFiles(files) {
    for (const f of files) {
        if (f.name.endsWith('.pdf')) {
            selectedFiles.push(f);
        }
    }
    renderFileList();
}

function renderFileList() {
    els.fileList.innerHTML = '';
    selectedFiles.forEach((f, i) => {
        const div = document.createElement('div');
        div.className = 'file-item';
        div.innerHTML = `
            <span>📄</span>
            <span class="file-name">${f.name}</span>
            <span class="file-size">${(f.size / 1024).toFixed(0)} KB</span>
            <button class="file-remove" data-index="${i}">✕</button>`;
        els.fileList.appendChild(div);
    });
    els.processBtn.style.display = selectedFiles.length > 0 ? 'inline-flex' : 'none';

    $$('.file-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            selectedFiles.splice(parseInt(btn.dataset.index), 1);
            renderFileList();
        });
    });
}

async function processLectures() {
    if (!currentSubject || selectedFiles.length === 0) return;

    els.processLog.style.display = 'block';
    els.processBtn.disabled = true;
    els.logEntry.textContent = '📤 Uploading files...';
    els.progressFill.style.width = '20%';

    // Upload files
    try {
        const formData = new FormData();
        selectedFiles.forEach(f => formData.append('files', f));
        await fetch(`${API}/api/subjects/${currentSubject}/upload`, {
            method: 'POST',
            body: formData,
        });
    } catch (e) {
        els.logEntry.textContent = '❌ Upload failed: ' + e.message;
        els.processBtn.disabled = false;
        return;
    }

    els.logEntry.textContent = '⚙️ Processing lectures (this may take a few minutes)...';
    els.progressFill.style.width = '50%';

    try {
        const resp = await fetch(`${API}/api/subjects/${currentSubject}/process`, { method: 'POST' });
        const data = await resp.json();

        if (resp.ok) {
            els.logEntry.textContent = `✅ Done! ${data.pages} pages → ${data.chunks} chunks → ${data.vectors} vectors`;
            els.progressFill.style.width = '100%';
            els.statVectors.textContent = data.vectors;
            selectedFiles = [];
            renderFileList();
            setStatus('online', `${data.vectors} vectors`);
            els.welcomeScreen.style.display = 'none';
            els.chatInputArea.style.display = 'block';
            loadLectures();
        } else {
            els.logEntry.textContent = '❌ Error: ' + (data.detail || 'Processing failed');
        }
    } catch (e) {
        els.logEntry.textContent = '❌ Error: ' + e.message;
    }

    els.processBtn.disabled = false;
}

// ============ HELPERS ============
function setStatus(state, text) {
    const dot = els.connectionStatus.querySelector('.status-dot');
    dot.className = `status-dot ${state}`;
    els.connectionStatus.childNodes[els.connectionStatus.childNodes.length - 1].textContent = ' ' + text;
}

function showLoading(text = 'Loading...') {
    els.loadingText.textContent = text;
    els.loadingOverlay.classList.add('active');
}

function hideLoading() {
    els.loadingOverlay.classList.remove('active');
}
