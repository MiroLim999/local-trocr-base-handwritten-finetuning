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
  model: null,     // selected OCR model key (see /models)
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
   OCR model selection
   ===================================================================== */
async function loadModels(preferKey = null) {
  const sel = $('modelSelect');
  try {
    const res = await fetch(`${CFG.OCR_API}/models`);
    const data = await res.json();
    const models = data.models || [];

    sel.innerHTML = '';
    models.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m.key;
      opt.textContent = m.available ? m.label : `${m.label} (not installed)`;
      opt.disabled = !m.available;
      sel.appendChild(opt);
    });

    // Prefer an explicit key, then the API default, then the first available.
    const firstAvailable = models.find((m) => m.available);
    const preferAvailable = preferKey && models.some((m) => m.key === preferKey && m.available);
    state.model = preferAvailable
      ? preferKey
      : data.default || (firstAvailable ? firstAvailable.key : null);
    if (state.model) sel.value = state.model;
  } catch (e) {
    sel.innerHTML = '<option value="">OCR engine offline</option>';
    state.model = null;
  }
  refreshModelActions();
}

$('modelSelect').addEventListener('change', (e) => {
  state.model = e.target.value || null;
  refreshModelActions();
  toast(`Model set to ${e.target.selectedOptions[0].textContent}`);
});

$('rescanModelsBtn').addEventListener('click', async () => {
  await loadModels(state.model);
  toast('Model list refreshed.');
});

// Keep the Delete/Rename buttons disabled while the (non-editable) base model is selected.
function refreshModelActions() {
  const disabled = !state.model || state.model === 'base';
  ['deleteModelBtn', 'renameModelBtn'].forEach((id) => {
    const btn = $(id);
    btn.disabled = disabled;
    btn.classList.toggle('opacity-40', disabled);
    btn.classList.toggle('cursor-not-allowed', disabled);
  });
}

$('deleteModelBtn').addEventListener('click', async () => {
  const key = state.model;
  if (!key || key === 'base') {
    return toast('The base model cannot be deleted.', 'error');
  }
  const sel = $('modelSelect');
  const label = sel.selectedOptions[0] ? sel.selectedOptions[0].textContent : key;

  const ok = await showConfirm({
    title: 'Delete model',
    message: `Delete "${label}"? This permanently removes the folder Models/${key} from disk and cannot be undone.`,
    okLabel: 'Delete',
  });
  if (!ok) return;

  try {
    const res = await fetch(`${CFG.OCR_API}/delete_model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: key }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
    toast(`Deleted "${label}".`, 'success');
    await loadModels();  // refresh list; falls back to the default model
  } catch (err) {
    console.error(err);
    toast(`Could not delete: ${err.message}`, 'error');
  }
});

$('renameModelBtn').addEventListener('click', async () => {
  const key = state.model;
  if (!key || key === 'base') {
    return toast('The base model cannot be renamed.', 'error');
  }

  const trimmed = await showRename(key);
  if (trimmed === null) return;              // cancelled
  if (!trimmed || trimmed === key) return;   // empty or unchanged

  try {
    const res = await fetch(`${CFG.OCR_API}/rename_model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: key, newName: trimmed }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
    toast(`Renamed to "${data.name}".`, 'success');
    await loadModels(data.name);  // refresh and select the renamed model
  } catch (err) {
    console.error(err);
    toast(`Could not rename: ${err.message}`, 'error');
  }
});

/* ---- Reusable modal helpers (styled replacements for prompt/confirm) ---- */
function _openModal(id) {
  const m = $(id);
  m.classList.remove('hidden');
  m.classList.add('flex');
}
function _closeModal(id) {
  const m = $(id);
  m.classList.add('hidden');
  m.classList.remove('flex');
}

// Promise-based confirm dialog. Resolves true (confirmed) / false (cancelled).
function showConfirm({ title = 'Are you sure?', message = '', okLabel = 'Confirm' } = {}) {
  return new Promise((resolve) => {
    $('confirmTitle').textContent = title;
    $('confirmMessage').textContent = message;
    $('confirmOk').textContent = okLabel;
    _openModal('confirmModal');

    const done = (result) => {
      _closeModal('confirmModal');
      $('confirmOk').onclick = null;
      $('confirmCancel').onclick = null;
      $('confirmClose').onclick = null;
      resolve(result);
    };
    $('confirmOk').onclick = () => done(true);
    $('confirmCancel').onclick = () => done(false);
    $('confirmClose').onclick = () => done(false);
  });
}

// Promise-based rename dialog. Resolves the new name, or null if cancelled.
function showRename(oldName) {
  return new Promise((resolve) => {
    $('renameOldName').textContent = oldName;
    const input = $('renameInput');
    input.value = oldName;
    _openModal('renameModal');
    input.focus();
    input.select();

    const done = (result) => {
      _closeModal('renameModal');
      $('renameSubmit').onclick = null;
      $('renameCancel').onclick = null;
      $('renameClose').onclick = null;
      input.onkeydown = null;
      resolve(result);
    };
    $('renameSubmit').onclick = () => done(input.value.trim());
    $('renameCancel').onclick = () => done(null);
    $('renameClose').onclick = () => done(null);
    input.onkeydown = (e) => {
      if (e.key === 'Enter') done(input.value.trim());
      else if (e.key === 'Escape') done(null);
    };
  });
}

/* =====================================================================
   Add a model — upload a folder of model files into Models/<name>/
   ===================================================================== */
const REQUIRED_MODEL_FILES = ['config.json'];
const REQUIRED_WEIGHTS = ['model.safetensors', 'pytorch_model.bin'];

function openAddModel() {
  $('addModelName').value = '';
  $('addModelFiles').value = '';
  $('addModelFileList').innerHTML = '';
  $('addModelSummary').classList.add('hidden');
  $('addModelDropTitle').textContent = 'Choose model folder';
  $('addModelProgress').classList.add('hidden');
  $('addModelBar').style.width = '0%';
  const modal = $('addModelModal');
  modal.classList.remove('hidden');
  modal.classList.add('flex');
}

function closeAddModel() {
  const modal = $('addModelModal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');
}

$('addModelBtn').addEventListener('click', openAddModel);
$('addModelClose').addEventListener('click', closeAddModel);
$('addModelCancel').addEventListener('click', closeAddModel);

// Preview the picked files and flag whether the required ones are present.
$('addModelFiles').addEventListener('change', (e) => {
  const list = $('addModelFileList');
  const files = [...e.target.files];
  list.innerHTML = '';

  if (!files.length) {
    $('addModelSummary').classList.add('hidden');
    return;
  }

  // Suggest a name from the selected folder if the field is empty.
  const folderName = files[0].webkitRelativePath
    ? files[0].webkitRelativePath.split('/')[0]
    : '';
  if (!$('addModelName').value && folderName) {
    $('addModelName').value = folderName;
  }
  $('addModelDropTitle').textContent = folderName || `${files.length} files selected`;

  const names = files.map((f) => f.name);
  const hasConfig = REQUIRED_MODEL_FILES.every((n) => names.includes(n));
  const hasWeights = REQUIRED_WEIGHTS.some((n) => names.includes(n));
  const valid = hasConfig && hasWeights;

  // One row per file: name on the left, size on the right.
  files.forEach((f) => {
    const li = document.createElement('li');
    li.className = 'flex items-center justify-between gap-3';
    const mb = f.size / (1024 * 1024);
    const size = mb >= 1 ? `${mb.toFixed(1)} MB` : `${(f.size / 1024).toFixed(0)} KB`;
    li.innerHTML =
      `<span class="font-mono text-slate-700 truncate">${escapeHtml(f.name)}</span>` +
      `<span class="text-slate-400 shrink-0">${size}</span>`;
    list.appendChild(li);
  });

  // Validation badge + total-size meta.
  const totalMb = files.reduce((a, f) => a + f.size, 0) / (1024 * 1024);
  const badge = $('addModelBadge');
  badge.className =
    'inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full ' +
    (valid ? 'bg-brand-100 text-brand-700' : 'bg-red-100 text-red-600');
  badge.textContent = valid ? '✓ Valid model folder' : '⚠ Missing required files';
  $('addModelMeta').textContent = `${files.length} files · ${totalMb.toFixed(0)} MB`;
  $('addModelSummary').classList.remove('hidden');
});

$('addModelSubmit').addEventListener('click', () => {
  const name = $('addModelName').value.trim();
  const files = [...$('addModelFiles').files];

  if (!name) return toast('Enter a model name.', 'error');
  if (!files.length) return toast('Select the model folder.', 'error');

  const names = files.map((f) => f.name);
  if (!REQUIRED_MODEL_FILES.every((n) => names.includes(n))) {
    return toast('The folder is missing config.json.', 'error');
  }
  if (!REQUIRED_WEIGHTS.some((n) => names.includes(n))) {
    return toast('The folder is missing model.safetensors.', 'error');
  }

  const form = new FormData();
  form.append('name', name);
  files.forEach((f) => form.append('files', f, f.name));

  const progress = $('addModelProgress');
  const bar = $('addModelBar');
  const status = $('addModelStatus');
  progress.classList.remove('hidden');
  status.textContent = 'Uploading…';
  $('addModelSubmit').disabled = true;

  const xhr = new XMLHttpRequest();
  // Upload to the Python API — it streams large weights to disk (Apache/PHP
  // buffers the whole upload in memory and chokes on multi-GB model files).
  xhr.open('POST', `${CFG.OCR_API}/add_model`);

  xhr.upload.onprogress = (ev) => {
    if (ev.lengthComputable) {
      const pct = Math.round((ev.loaded / ev.total) * 100);
      bar.style.width = pct + '%';
      status.textContent = `Uploading… ${pct}%`;
    }
  };

  xhr.onload = async () => {
    $('addModelSubmit').disabled = false;
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch (_) {}
    if (xhr.status >= 200 && xhr.status < 300 && data.ok) {
      status.textContent = 'Done.';
      toast(`Model "${data.name}" added.`, 'success');
      closeAddModel();
      await loadModels(data.name);   // refresh list and select the new model
    } else {
      status.textContent = data.error || `Upload failed (HTTP ${xhr.status}).`;
      toast(data.error || 'Could not add the model.', 'error');
    }
  };

  xhr.onerror = () => {
    $('addModelSubmit').disabled = false;
    status.textContent = 'Network error during upload.';
    toast('Upload failed — is the Python API running?', 'error');
  };

  xhr.send(form);
});

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
      body: JSON.stringify({ fields: payload, model: state.model }),
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
loadModels();
checkApi();
loadSavedDocs();
setInterval(checkApi, 15000);
