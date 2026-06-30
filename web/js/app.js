/* =====================================================================
   Civil Records Digitizer — frontend logic
   - Renders a scanned PDF with PDF.js
   - Lets the user place / drag / resize field boxes
   - Crops each field and sends it to the Python TrOCR API
   - Shows CER / WER / exact-match metrics for human verification
   - Saves the verified document as a PNG via PHP
   ===================================================================== */

const CFG = window.APP_CONFIG;

// PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc =
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

// ---- Application state ----
const state = {
  docType: null,
  pdfDoc: null,
  page: null,
  fields: [],      // [{ id, name, el }]
  results: [],     // [{ name, text, confidence, cropDataUrl }]
  fieldSeq: 0,
};

// ---- Element helpers ----
const $ = (id) => document.getElementById(id);
const views = {
  upload:  $('view-upload'),
  fields:  $('view-fields'),
  results: $('view-results'),
  saved:   $('view-saved'),
};

function showView(name) {
  Object.values(views).forEach((v) => v.classList.add('hidden'));
  views[name].classList.remove('hidden');
}

function toast(msg, kind = 'info') {
  const el = $('toast');
  el.textContent = msg;
  el.className =
    'fixed bottom-6 right-6 px-4 py-3 rounded-lg shadow-lg text-sm font-medium ' +
    (kind === 'error'
      ? 'bg-red-600 text-white'
      : kind === 'success'
      ? 'bg-brand-700 text-white'
      : 'bg-slate-800 text-white');
  el.classList.remove('hidden');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add('hidden'), 3200);
}

/* =====================================================================
   OCR API health
   ===================================================================== */
async function checkApi() {
  const dot = $('apiDot');
  const label = $('apiStatus');
  try {
    const res = await fetch(`${CFG.OCR_API}/health`);
    const data = await res.json();
    dot.className = 'h-2 w-2 rounded-full bg-brand-500';
    label.textContent = `OCR engine ready · ${data.device || ''}`.trim();
  } catch (e) {
    dot.className = 'h-2 w-2 rounded-full bg-red-500';
    label.textContent = 'OCR engine offline — start the Python API';
  }
}

/* =====================================================================
   Document type selection
   ===================================================================== */
document.querySelectorAll('.doctype-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.doctype-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    state.docType = btn.dataset.doctype;
  });
});

/* =====================================================================
   Dropzone / file input
   ===================================================================== */
const dropzone = $('dropzone');
const fileInput = $('fileInput');

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

async function handleFile(file) {
  if (!state.docType) {
    toast('Pick a document type first.', 'error');
    return;
  }

  const isPdf = file.type === 'application/pdf' || /\.pdf$/i.test(file.name);
  const isImage = file.type.startsWith('image/') ||
    /\.(png|jpe?g|webp|bmp|tiff?|gif)$/i.test(file.name);

  if (!isPdf && !isImage) {
    toast('Please upload a PDF or image file.', 'error');
    return;
  }

  try {
    if (isPdf) {
      const buf = await file.arrayBuffer();
      state.pdfDoc = await pdfjsLib.getDocument({ data: buf }).promise;
      await renderPage(1);
    } else {
      state.pdfDoc = null;
      await renderImage(file);
    }
    initTemplateFields();
    showView('fields');
  } catch (err) {
    console.error(err);
    toast('Could not read that file.', 'error');
  }
}

// Draw an uploaded image onto the document canvas at its native resolution.
function renderImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const canvas = $('pdfCanvas');
      const ctx = canvas.getContext('2d');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      resolve();
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('image load failed'));
    };
    img.src = url;
  });
}

/* =====================================================================
   PDF rendering
   ===================================================================== */
async function renderPage(pageNum) {
  const page = await state.pdfDoc.getPage(pageNum);
  state.page = page;

  const canvas = $('pdfCanvas');
  const ctx = canvas.getContext('2d');

  // Render at 2x for crisp crops; CSS scales it down to fit the column.
  const scale = 2.0;
  const viewport = page.getViewport({ scale });
  canvas.width = viewport.width;
  canvas.height = viewport.height;

  await page.render({ canvasContext: ctx, viewport }).promise;
}

/* =====================================================================
   Field boxes (auto-placed template, draggable + resizable)
   ===================================================================== */
const overlay = $('overlay');

function initTemplateFields() {
  overlay.innerHTML = '';
  state.fields = [];
  state.fieldSeq = 0;

  const template = CFG.FIELD_TEMPLATES[state.docType] || [];
  const cw = $('pdfCanvas').clientWidth;
  const ch = $('pdfCanvas').clientHeight;

  template.forEach((f) => {
    addField(f.name, f.x * cw, f.y * ch, f.w * cw, f.h * ch);
  });
  renderLegend();
}

function addField(name, left, top, width, height) {
  const id = ++state.fieldSeq;
  const box = document.createElement('div');
  box.className = 'field-box';
  box.style.left = `${left}px`;
  box.style.top = `${top}px`;
  box.style.width = `${width}px`;
  box.style.height = `${height}px`;
  box.dataset.id = id;

  const tag = document.createElement('span');
  tag.className = 'field-tag';
  tag.textContent = name;

  const remove = document.createElement('span');
  remove.className = 'field-remove';
  remove.textContent = '×';
  remove.title = 'Remove field';
  remove.addEventListener('mousedown', (e) => e.stopPropagation());
  remove.addEventListener('click', (e) => {
    e.stopPropagation();
    box.remove();
    state.fields = state.fields.filter((x) => x.id !== id);
    renderLegend();
  });

  const handle = document.createElement('div');
  handle.className = 'resize-handle';

  box.append(tag, remove, handle);
  overlay.appendChild(box);

  makeInteractive(box, handle);
  state.fields.push({ id, name, el: box });
  return box;
}

function makeInteractive(box, handle) {
  let mode = null;
  let startX, startY, startLeft, startTop, startW, startH;

  const onDown = (e, m) => {
    e.preventDefault();
    mode = m;
    startX = e.clientX;
    startY = e.clientY;
    startLeft = box.offsetLeft;
    startTop = box.offsetTop;
    startW = box.offsetWidth;
    startH = box.offsetHeight;
    document.querySelectorAll('.field-box').forEach((b) => b.classList.remove('selected'));
    box.classList.add('selected');
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };

  const onMove = (e) => {
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    const maxW = overlay.clientWidth;
    const maxH = overlay.clientHeight;

    if (mode === 'move') {
      let l = Math.min(Math.max(0, startLeft + dx), maxW - box.offsetWidth);
      let t = Math.min(Math.max(0, startTop + dy), maxH - box.offsetHeight);
      box.style.left = `${l}px`;
      box.style.top = `${t}px`;
    } else if (mode === 'resize') {
      let w = Math.max(24, Math.min(startW + dx, maxW - box.offsetLeft));
      let h = Math.max(16, Math.min(startH + dy, maxH - box.offsetTop));
      box.style.width = `${w}px`;
      box.style.height = `${h}px`;
    }
  };

  const onUp = () => {
    mode = null;
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
  };

  box.addEventListener('mousedown', (e) => onDown(e, 'move'));
  handle.addEventListener('mousedown', (e) => {
    e.stopPropagation();
    onDown(e, 'resize');
  });
}

function renderLegend() {
  const ul = $('fieldLegend');
  ul.innerHTML = '';
  if (!state.fields.length) {
    ul.innerHTML = '<li class="text-slate-400">No fields. Click “Add field”.</li>';
    return;
  }
  state.fields.forEach((f) => {
    const li = document.createElement('li');
    li.className = 'flex items-center gap-2 text-slate-600';
    li.innerHTML =
      '<span class="h-2.5 w-2.5 rounded-sm" style="background:#0d9488"></span>' +
      `<span>${escapeHtml(f.name)}</span>`;
    ul.appendChild(li);
  });
}

// Add a custom field
$('addFieldBtn').addEventListener('click', () => {
  const name = prompt('Field name (e.g. Registry No.)');
  if (!name) return;
  const cw = $('pdfCanvas').clientWidth;
  addField(name.trim(), cw * 0.3, cw * 0.06, cw * 0.35, cw * 0.05);
  renderLegend();
});

$('backToUpload').addEventListener('click', () => showView('upload'));

/* =====================================================================
   Scan now — crop each field and call the OCR API
   ===================================================================== */
$('scanNowBtn').addEventListener('click', runScan);

function cropField(box) {
  const canvas = $('pdfCanvas');
  const ratio = canvas.width / canvas.clientWidth; // source px per display px

  const sx = box.offsetLeft * ratio;
  const sy = box.offsetTop * ratio;
  const sw = box.offsetWidth * ratio;
  const sh = box.offsetHeight * ratio;

  const out = document.createElement('canvas');
  out.width = Math.max(1, Math.round(sw));
  out.height = Math.max(1, Math.round(sh));
  const octx = out.getContext('2d');
  octx.fillStyle = '#ffffff';
  octx.fillRect(0, 0, out.width, out.height);
  octx.drawImage(canvas, sx, sy, sw, sh, 0, 0, out.width, out.height);
  return out.toDataURL('image/png');
}

async function runScan() {
  if (!state.fields.length) {
    toast('Add at least one field first.', 'error');
    return;
  }

  const payload = state.fields.map((f) => ({
    name: f.name,
    image: cropField(f.el),
  }));

  const scanOverlay = $('scanOverlay');
  scanOverlay.style.display = 'flex';
  $('scanProgress').textContent = `${payload.length} field(s) queued`;

  try {
    const res = await fetch(`${CFG.OCR_API}/ocr`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fields: payload }),
    });
    if (!res.ok) throw new Error(`API ${res.status}`);
    const data = await res.json();

    state.results = data.results.map((r, i) => ({
      name: r.name,
      text: r.text || '',
      verified: r.text || '',
      confidence: typeof r.confidence === 'number' ? r.confidence : 0,
      cropDataUrl: payload[i].image,
    }));

    buildResults();
    showView('results');
  } catch (err) {
    console.error(err);
    toast('OCR failed — is the Python API running?', 'error');
  } finally {
    scanOverlay.style.display = 'none';
  }
}

/* =====================================================================
   Results / verification view
   ===================================================================== */
function buildResults() {
  const wrap = $('compareRows');
  wrap.innerHTML = '';

  state.results.forEach((r, idx) => {
    const row = document.createElement('div');
    row.className = 'compare-row';

    const confClass =
      r.confidence >= 90 ? 'conf-high' : r.confidence >= 75 ? 'conf-mid' : 'conf-low';

    row.innerHTML = `
      <div class="px-4 py-3 text-sm font-medium text-slate-700 flex items-center">
        ${escapeHtml(r.name)}
      </div>
      <div class="px-4 py-3 border-l border-slate-200">
        <img src="${r.cropDataUrl}" class="crop-img mb-2" alt="crop" />
        <div class="text-sm text-slate-600">
          <span class="font-mono">${escapeHtml(r.text) || '<span class="text-slate-400">(blank)</span>'}</span>
          <span class="conf-pill ${confClass} ml-2">${r.confidence.toFixed(0)}%</span>
        </div>
      </div>
      <div class="px-4 py-3 border-l border-slate-200 flex items-center">
        <input class="verify-input" data-idx="${idx}" value="${escapeAttr(r.verified)}" />
      </div>
    `;
    wrap.appendChild(row);
  });

  wrap.querySelectorAll('.verify-input').forEach((input) => {
    input.addEventListener('input', (e) => {
      const i = +e.target.dataset.idx;
      state.results[i].verified = e.target.value;
      updateMetrics();
    });
  });

  updateMetrics();
}

$('backToFields').addEventListener('click', () => showView('fields'));

/* =====================================================================
   Metrics — CER / WER / exact match (ported from metrics.py)
   reference = verified (human) text, hypothesis = AI prediction
   ===================================================================== */
function levenshtein(ref, hyp) {
  const m = ref.length, n = hyp.length;
  if (m === 0) return n;
  if (n === 0) return m;
  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  for (let i = 1; i <= m; i++) {
    const curr = [i];
    for (let j = 1; j <= n; j++) {
      const cost = ref[i - 1] === hyp[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    prev = curr;
  }
  return prev[n];
}

function computeMetrics() {
  let charErr = 0, chars = 0, wordErr = 0, words = 0, exact = 0;
  state.results.forEach((r) => {
    const ref = r.verified.trim();
    const hyp = r.text.trim();
    charErr += levenshtein([...ref], [...hyp]);
    chars += ref.length;
    const rw = ref.split(/\s+/).filter(Boolean);
    const hw = hyp.split(/\s+/).filter(Boolean);
    wordErr += levenshtein(rw, hw);
    words += rw.length;
    if (ref === hyp) exact++;
  });
  return {
    cer: chars ? charErr / chars : 0,
    wer: words ? wordErr / words : 0,
    exact,
    total: state.results.length,
  };
}

function updateMetrics() {
  const m = computeMetrics();
  $('m-cer').textContent = (m.cer * 100).toFixed(2) + '%';
  $('m-wer').textContent = (m.wer * 100).toFixed(2) + '%';
  $('m-exact').textContent = `${m.exact}/${m.total}`;

  const confs = state.results.map((r) => r.confidence).filter((c) => c > 0);
  $('m-conf').textContent = confs.length
    ? (confs.reduce((a, b) => a + b, 0) / confs.length).toFixed(1) + '%'
    : '—';
}

/* =====================================================================
   Save verified document as PNG (rendered + sent to PHP)
   ===================================================================== */
$('saveDocBtn').addEventListener('click', saveDocument);

function renderDocumentPng() {
  const pad = 48;
  const width = 900;
  const rowH = 64;
  const headerH = 150;
  const m = computeMetrics();
  const height = headerH + state.results.length * rowH + 120;

  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  const ctx = c.getContext('2d');

  // Background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  // Header bar
  ctx.fillStyle = '#0f766e';
  ctx.fillRect(0, 0, width, 8);

  ctx.fillStyle = '#0f172a';
  ctx.font = '700 26px Inter, sans-serif';
  ctx.fillText(CFG.DOC_LABELS[state.docType] || 'Document', pad, 56);

  ctx.fillStyle = '#64748b';
  ctx.font = '400 14px Inter, sans-serif';
  ctx.fillText('Digitized record · ' + new Date().toLocaleString(), pad, 82);
  ctx.fillText(
    `CER ${(m.cer * 100).toFixed(2)}%   ·   WER ${(m.wer * 100).toFixed(2)}%   ·   Exact ${m.exact}/${m.total}`,
    pad, 104
  );

  // Divider
  ctx.strokeStyle = '#e2e8f0';
  ctx.beginPath();
  ctx.moveTo(pad, headerH - 20);
  ctx.lineTo(width - pad, headerH - 20);
  ctx.stroke();

  // Fields
  let y = headerH + 10;
  state.results.forEach((r) => {
    ctx.fillStyle = '#94a3b8';
    ctx.font = '600 12px Inter, sans-serif';
    ctx.fillText(r.name.toUpperCase(), pad, y);

    ctx.fillStyle = '#0f172a';
    ctx.font = '500 20px Inter, sans-serif';
    ctx.fillText(r.verified || '—', pad, y + 26);

    ctx.strokeStyle = '#f1f5f9';
    ctx.beginPath();
    ctx.moveTo(pad, y + 40);
    ctx.lineTo(width - pad, y + 40);
    ctx.stroke();

    y += rowH;
  });

  ctx.fillStyle = '#94a3b8';
  ctx.font = '400 12px Inter, sans-serif';
  ctx.fillText('Generated by Civil Records Digitizer (TrOCR fine-tuned)', pad, height - 28);

  return c.toDataURL('image/png');
}

async function saveDocument() {
  if (!state.results.length) return;

  const m = computeMetrics();
  const nameField = state.results.find((r) => /name/i.test(r.name));
  const titlePart = nameField && nameField.verified ? ' - ' + nameField.verified : '';
  const title = (CFG.DOC_LABELS[state.docType] || 'Document') + titlePart;

  const image = renderDocumentPng();

  try {
    const res = await fetch('save_doc.php', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        docType: state.docType,
        image,
        fields: state.results.map((r) => ({ name: r.name, value: r.verified })),
        metrics: { cer: m.cer, wer: m.wer, exact: m.exact, total: m.total },
      }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'save failed');
    toast('Saved to your documents.', 'success');
    await loadSavedDocs();
  } catch (err) {
    console.error(err);
    toast('Could not save the document.', 'error');
  }
}

/* =====================================================================
   Saved documents (sidebar list + viewer)
   ===================================================================== */
async function loadSavedDocs() {
  try {
    const res = await fetch('list_docs.php');
    const data = await res.json();
    renderSavedList(data.docs || []);
  } catch (e) {
    // PHP not running — leave the empty state.
  }
}

function renderSavedList(docs) {
  const list = $('savedList');
  $('savedCount').textContent = docs.length;
  list.innerHTML = '';

  if (!docs.length) {
    list.innerHTML =
      '<p class="px-3 py-6 text-sm text-slate-400 text-center">No saved documents yet.</p>';
    return;
  }

  docs.forEach((d) => {
    const btn = document.createElement('button');
    btn.className = 'saved-item';
    btn.innerHTML = `
      <span class="block text-sm font-medium text-slate-700 truncate">${escapeHtml(d.title)}</span>
      <span class="block text-xs text-slate-400 mt-0.5">${escapeHtml(CFG.DOC_LABELS[d.docType] || d.docType)}</span>
    `;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.saved-item').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      openSaved(d);
    });
    list.appendChild(btn);
  });
}

function openSaved(d) {
  $('savedTitle').textContent = d.title;
  const parts = [CFG.DOC_LABELS[d.docType] || d.docType];
  if (d.savedAt) parts.push(new Date(d.savedAt).toLocaleString());
  if (d.metrics) {
    parts.push(
      `CER ${(d.metrics.cer * 100).toFixed(2)}% · WER ${(d.metrics.wer * 100).toFixed(2)}%`
    );
  }
  $('savedMeta').textContent = parts.join('  ·  ');
  $('savedImg').src = d.url;
  const dl = $('downloadSaved');
  dl.href = d.url;
  dl.setAttribute('download', d.file);
  $('deleteSaved').onclick = () => deleteSaved(d.file);
  showView('saved');
}

async function deleteSaved(file) {
  if (!confirm('Delete this saved document?')) return;
  try {
    await fetch('delete_doc.php', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file }),
    });
    toast('Document deleted.', 'success');
    await loadSavedDocs();
    startNew();
  } catch (e) {
    toast('Could not delete.', 'error');
  }
}

/* =====================================================================
   Navigation / init
   ===================================================================== */
function startNew() {
  document.querySelectorAll('.saved-item').forEach((b) => b.classList.remove('active'));
  showView('upload');
}

document.querySelector('[data-nav="new"]').addEventListener('click', startNew);

// small HTML escaping helpers
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}
function escapeAttr(s) {
  return String(s).replace(/"/g, '&quot;');
}

// Boot
checkApi();
loadSavedDocs();
setInterval(checkApi, 15000);
