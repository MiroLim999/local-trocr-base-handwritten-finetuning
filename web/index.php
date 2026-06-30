<?php
/**
 * index.php
 * Single-page dashboard for the TrOCR civil-document digitizer.
 * PHP only serves this file and the saved-docs storage endpoints; the
 * OCR work is done by the Python Flask API (see ../api/app.py).
 */
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Civil Records Digitizer — TrOCR</title>

  <!-- Tailwind (CDN build for prototyping) -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
          },
          colors: {
            brand: {
              50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4',
              400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e',
              800: '#115e59', 900: '#134e4a',
            },
          },
        },
      },
    };
  </script>

  <!-- Inter font -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />

  <!-- PDF.js for rendering scanned PDFs in the browser -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>

  <link rel="stylesheet" href="css/styles.css" />
</head>
<body class="bg-slate-100 text-slate-800 font-sans antialiased">

  <div class="flex h-screen overflow-hidden">

    <!-- ============================ SIDEBAR ============================ -->
    <aside class="w-72 shrink-0 bg-white border-r border-slate-200 flex flex-col">
      <div class="px-5 py-5 border-b border-slate-200">
        <div class="flex items-center gap-3">
          <div class="h-9 w-9 rounded-lg bg-brand-700 flex items-center justify-center text-white font-bold">CR</div>
          <div>
            <h1 class="text-sm font-semibold leading-tight">Civil Records</h1>
            <p class="text-xs text-slate-500 leading-tight">Handwriting Digitizer</p>
          </div>
        </div>
      </div>

      <nav class="px-3 py-4 space-y-1">
        <button data-nav="new" class="nav-btn active w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-left">
          <svg class="h-5 w-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 4v16m8-8H4"/></svg>
          New Document
        </button>
      </nav>

      <div class="px-5 pt-2 pb-2 flex items-center justify-between">
        <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400">Saved Documents</h2>
        <span id="savedCount" class="text-xs text-slate-400">0</span>
      </div>

      <div id="savedList" class="flex-1 overflow-y-auto px-3 pb-4 space-y-1">
        <p class="px-3 py-6 text-sm text-slate-400 text-center">No saved documents yet.</p>
      </div>

      <div class="px-5 py-3 border-t border-slate-200 flex items-center gap-2 text-xs">
        <span id="apiDot" class="h-2 w-2 rounded-full bg-slate-300"></span>
        <span id="apiStatus" class="text-slate-500">Checking OCR engine…</span>
      </div>
    </aside>

    <!-- ============================ MAIN ============================ -->
    <main class="flex-1 overflow-y-auto">
      <div class="max-w-6xl mx-auto px-8 py-8">

        <!-- ---------- VIEW: UPLOAD ---------- -->
        <section id="view-upload" class="view">
          <header class="mb-6">
            <h2 class="text-2xl font-semibold text-slate-900">New Document</h2>
            <p class="text-slate-500 mt-1">Upload a scanned certificate, mark its fields, then convert the handwriting to text.</p>
          </header>

          <div class="bg-white rounded-xl border border-slate-200 p-6">
            <label class="block text-sm font-medium text-slate-700 mb-2">Document type</label>
            <div id="docTypePicker" class="grid grid-cols-3 gap-3 mb-6">
              <button data-doctype="birth" class="doctype-btn rounded-lg border border-slate-200 px-4 py-4 text-left">
                <span class="block text-sm font-semibold text-slate-800">Birth</span>
                <span class="block text-xs text-slate-500 mt-0.5">Certificate</span>
              </button>
              <button data-doctype="death" class="doctype-btn rounded-lg border border-slate-200 px-4 py-4 text-left">
                <span class="block text-sm font-semibold text-slate-800">Death</span>
                <span class="block text-xs text-slate-500 mt-0.5">Certificate</span>
              </button>
              <button data-doctype="marriage" class="doctype-btn rounded-lg border border-slate-200 px-4 py-4 text-left">
                <span class="block text-sm font-semibold text-slate-800">Marriage</span>
                <span class="block text-xs text-slate-500 mt-0.5">Certificate</span>
              </button>
            </div>

            <div id="dropzone" class="dropzone rounded-xl border-2 border-dashed border-slate-300 px-6 py-14 text-center cursor-pointer transition">
              <svg class="h-10 w-10 mx-auto text-slate-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/></svg>
              <p class="mt-3 text-slate-700 font-medium">Drag &amp; drop a scanned certificate here</p>
              <p class="text-sm text-slate-500">or click to browse — PDF or image (PNG, JPG, WEBP, BMP, TIFF)</p>
              <input id="fileInput" type="file" accept="application/pdf,image/*" class="hidden" />
            </div>
          </div>
        </section>

        <!-- ---------- VIEW: FIELD DETECTION ---------- -->
        <section id="view-fields" class="view hidden">
          <header class="mb-6 flex items-start justify-between gap-4">
            <div>
              <h2 class="text-2xl font-semibold text-slate-900">Mark the fields</h2>
              <p class="text-slate-500 mt-1">Drag each box onto the matching handwritten value. Drag a corner to resize.</p>
            </div>
            <div class="flex items-center gap-2">
              <button id="backToUpload" class="btn-ghost">Back</button>
              <button id="addFieldBtn" class="btn-secondary">+ Add field</button>
              <button id="scanNowBtn" class="btn-primary">Scan now</button>
            </div>
          </header>

          <div class="grid grid-cols-[1fr_280px] gap-6">
            <div class="bg-white rounded-xl border border-slate-200 p-4">
              <div id="docStage" class="relative inline-block max-w-full">
                <canvas id="pdfCanvas" class="block max-w-full rounded-md shadow-sm"></canvas>
                <div id="overlay" class="absolute inset-0"></div>
              </div>
            </div>

            <aside class="bg-white rounded-xl border border-slate-200 p-4 h-fit">
              <h3 class="text-sm font-semibold text-slate-700 mb-3">Fields to capture</h3>
              <ul id="fieldLegend" class="space-y-2 text-sm"></ul>
              <p class="text-xs text-slate-400 mt-4 leading-relaxed">Tip: position each box tightly around the handwriting for the best accuracy.</p>
            </aside>
          </div>
        </section>

        <!-- ---------- VIEW: RESULTS / VERIFY ---------- -->
        <section id="view-results" class="view hidden">
          <header class="mb-6 flex items-start justify-between gap-4">
            <div>
              <h2 class="text-2xl font-semibold text-slate-900">Review &amp; verify</h2>
              <p class="text-slate-500 mt-1">Compare each crop with the AI reading and correct it. Metrics update as you edit.</p>
            </div>
            <div class="flex items-center gap-2">
              <button id="backToFields" class="btn-ghost">Back</button>
              <button id="saveDocBtn" class="btn-primary">Save as PNG</button>
            </div>
          </header>

          <!-- Metrics -->
          <div class="grid grid-cols-4 gap-4 mb-6">
            <div class="metric-card">
              <p class="metric-label">Character Error Rate</p>
              <p id="m-cer" class="metric-value">0.00%</p>
            </div>
            <div class="metric-card">
              <p class="metric-label">Word Error Rate</p>
              <p id="m-wer" class="metric-value">0.00%</p>
            </div>
            <div class="metric-card">
              <p class="metric-label">Exact-match Fields</p>
              <p id="m-exact" class="metric-value">0/0</p>
            </div>
            <div class="metric-card">
              <p class="metric-label">Avg. Confidence</p>
              <p id="m-conf" class="metric-value">—</p>
            </div>
          </div>

          <!-- Side-by-side comparison -->
          <div class="bg-white rounded-xl border border-slate-200 overflow-hidden">
            <div class="grid grid-cols-[200px_1fr_1fr] bg-slate-50 border-b border-slate-200 text-xs font-semibold uppercase tracking-wide text-slate-500">
              <div class="px-4 py-3">Field</div>
              <div class="px-4 py-3 border-l border-slate-200">Scanned crop · AI reading</div>
              <div class="px-4 py-3 border-l border-slate-200">Verified text (editable)</div>
            </div>
            <div id="compareRows"></div>
          </div>
        </section>

        <!-- ---------- VIEW: SAVED DOC VIEWER ---------- -->
        <section id="view-saved" class="view hidden">
          <header class="mb-6 flex items-start justify-between gap-4">
            <div>
              <h2 id="savedTitle" class="text-2xl font-semibold text-slate-900">Saved document</h2>
              <p id="savedMeta" class="text-slate-500 mt-1"></p>
            </div>
            <div class="flex items-center gap-2">
              <a id="downloadSaved" class="btn-secondary" download>Download PNG</a>
              <button id="deleteSaved" class="btn-danger">Delete</button>
            </div>
          </header>
          <div class="bg-white rounded-xl border border-slate-200 p-6 flex justify-center">
            <img id="savedImg" alt="Saved document" class="max-w-full rounded-md border border-slate-200" />
          </div>
        </section>

      </div>
    </main>
  </div>

  <!-- Toast -->
  <div id="toast" class="fixed bottom-6 right-6 hidden px-4 py-3 rounded-lg shadow-lg text-sm font-medium"></div>

  <!-- Scan overlay -->
  <div id="scanOverlay" class="fixed inset-0 bg-slate-900/40 hidden items-center justify-center z-50">
    <div class="bg-white rounded-xl px-8 py-6 shadow-xl text-center">
      <div class="mx-auto h-8 w-8 border-2 border-brand-600 border-t-transparent rounded-full animate-spin"></div>
      <p class="mt-4 text-sm font-medium text-slate-700">Reading handwriting with TrOCR…</p>
      <p id="scanProgress" class="mt-1 text-xs text-slate-400"></p>
    </div>
  </div>

  <script src="js/config.js"></script>
  <script src="js/app.js"></script>
</body>
</html>
