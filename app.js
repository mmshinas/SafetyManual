"use strict";

/* ---------------------------------------------------------------------
 * Offline search engine (BM25-lite over a precomputed inverted index).
 * Everything here reads only from data/*.json extracted from the source
 * manual — there is no generation step, so every result IS the manual.
 * ------------------------------------------------------------------- */

const STOPWORDS = new Set(
  "a an the and or but if then else of to in on at for with by from as is are was were be been being this that these those it its into over under above below not no nor so than too very can will just"
    .split(" ")
);

function tokenize(text) {
  return (text.toLowerCase().match(/[a-z0-9][a-z0-9\-]*/g) || []).filter(
    (t) => t.length > 1 || /[0-9]/.test(t)
  );
}

class SearchEngine {
  constructor(chunksData, indexData) {
    this.chunks = chunksData.chunks;
    this.meta = chunksData.meta;
    this.numChunks = indexData.numChunks;
    this.avgLen = indexData.avgLen || 1;
    this.chunkLengths = indexData.chunkLengths;
    this.postings = indexData.postings; // token -> [[chunkId, tf], ...]
    this.vocab = Object.keys(this.postings).sort();
  }

  // Expand a raw query token to itself plus any vocab entries it prefixes,
  // so partial words while typing still surface useful matches.
  expandToken(token) {
    if (this.postings[token]) return [token];
    const matches = [];
    // vocab is sorted; linear scan is plenty fast for manual-sized vocabularies
    for (const v of this.vocab) {
      if (v.startsWith(token)) matches.push(v);
      if (matches.length >= 6) break;
    }
    return matches;
  }

  search(query, limit = 25) {
    const rawTokens = tokenize(query).filter((t) => !STOPWORDS.has(t));
    if (rawTokens.length === 0) return { results: [], queryTokens: [] };

    const N = this.numChunks;
    const k1 = 1.5;
    const b = 0.75;
    const scores = new Map(); // chunkId -> score
    const matchedTermsByChunk = new Map(); // chunkId -> Set(term)

    const expandedTermSet = new Set();
    for (const rt of rawTokens) {
      for (const term of this.expandToken(rt)) expandedTermSet.add(term);
    }

    for (const term of expandedTermSet) {
      const postingList = this.postings[term];
      if (!postingList) continue;
      const df = postingList.length;
      const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
      for (const [chunkId, tf] of postingList) {
        const len = this.chunkLengths[chunkId] || 1;
        const denom = tf + k1 * (1 - b + (b * len) / this.avgLen);
        const score = idf * ((tf * (k1 + 1)) / denom);
        scores.set(chunkId, (scores.get(chunkId) || 0) + score);
        if (!matchedTermsByChunk.has(chunkId)) matchedTermsByChunk.set(chunkId, new Set());
        matchedTermsByChunk.get(chunkId).add(term);
      }
    }

    const ranked = Array.from(scores.entries())
      .map(([chunkId, score]) => {
        const matchedTerms = matchedTermsByChunk.get(chunkId);
        // Reward chunks that cover more of the distinct query terms.
        const coverage = matchedTerms.size / expandedTermSet.size;
        return { chunkId, score: score * (0.5 + 0.5 * coverage), matchedTerms };
      })
      .sort((a, b2) => b2.score - a.score)
      .slice(0, limit);

    const results = ranked.map(({ chunkId, matchedTerms }) => {
      const chunk = this.chunks[chunkId];
      return {
        chunk,
        snippet: buildSnippet(chunk.text, matchedTerms),
      };
    });

    return { results, queryTokens: Array.from(expandedTermSet) };
  }

  getChunk(id) {
    return this.chunks[id];
  }
}

function buildSnippet(text, matchedTerms, radius = 140) {
  const lower = text.toLowerCase();
  let hitIndex = -1;
  for (const term of matchedTerms) {
    const idx = lower.indexOf(term);
    if (idx !== -1 && (hitIndex === -1 || idx < hitIndex)) hitIndex = idx;
  }
  if (hitIndex === -1) hitIndex = 0;
  const start = Math.max(0, hitIndex - radius);
  const end = Math.min(text.length, hitIndex + radius);
  let snippet = text.slice(start, end).trim();
  if (start > 0) snippet = "…" + snippet;
  if (end < text.length) snippet = snippet + "…";
  return snippet;
}

function highlight(text, terms) {
  if (!terms || (terms instanceof Set ? terms.size === 0 : terms.length === 0)) {
    return escapeHtml(text);
  }
  const termList = Array.from(terms).sort((a, b) => b.length - a.length);
  const escaped = termList.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp("(" + escaped.join("|") + ")", "gi");
  return escapeHtml(text).replace(re, (m) => `<mark>${m}</mark>`);
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/* ---------------------------------------------------------------------
 * App wiring
 * ------------------------------------------------------------------- */

const state = {
  engine: null,
  config: null,
  lastResults: [],
  lastQueryTokens: [],
  fontScale: parseFloat(localStorage.getItem("manual.fontScale") || "1"),
  recent: JSON.parse(localStorage.getItem("manual.recent") || "[]"),
  theme: localStorage.getItem("manual.theme") || "system",
  pagesWithImages: new Set(),
};

const el = {};
function qs(id) {
  return document.getElementById(id);
}

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

async function boot() {
  cacheElements();
  document.documentElement.style.setProperty("--font-scale", state.fontScale);
  applyTheme(state.theme);
  wireEvents();
  updateOnlineStatus();
  window.addEventListener("online", updateOnlineStatus);
  window.addEventListener("offline", updateOnlineStatus);

  try {
    // A single-file preview build (e.g. published as an Artifact) can skip
    // network/service-worker entirely by pre-embedding the three data files.
    const preloaded = window.__MANUAL_PRELOADED__;
    const [config, chunksData, indexData] = preloaded
      ? [preloaded.config, preloaded.chunksData, preloaded.indexData]
      : await Promise.all([
          loadJSON("data/config.json"),
          loadJSON("data/chunks.json"),
          loadJSON("data/index.json"),
        ]);
    state.config = config;
    state.isPreview = !!preloaded;
    state.engine = new SearchEngine(chunksData, indexData);
    state.pagesWithImages = new Set(state.engine.meta.pagesWithImages || []);
    el.title.textContent = config.appTitle || "E-Manual Search";
    document.title = config.appTitle || "E-Manual Search";
    el.subtitle.textContent = config.subtitle || "";
    renderEmptyState();
    resetReader();
    qs("loading").remove();
  } catch (err) {
    qs("loading").innerHTML = `
      <div class="big-icon">⚠️</div>
      <div><strong>Couldn't load the manual data.</strong></div>
      <div style="font-size:0.85rem;max-width:280px;">${escapeHtml(String(err.message || err))}<br><br>
      Run the build script against your manual PDF, then reload.</div>`;
  }

  if (!state.isPreview) {
    registerServiceWorker();
    wireInstallPrompt();
  }
}

function cacheElements() {
  el.title = qs("app-title");
  el.subtitle = qs("app-subtitle");
  el.searchInput = qs("search-input");
  el.clearBtn = qs("clear-btn");
  el.results = qs("results");
  el.statusDot = qs("status-dot");
  el.statusText = qs("status-text");
  el.tocBtn = qs("toc-btn");
  el.settingsBtn = qs("settings-btn");
  el.reader = qs("reader");
  el.readerBody = qs("reader-body");
  el.readerTitle = qs("reader-title");
  el.readerClose = qs("reader-close");
  el.readerViewImage = qs("reader-view-image");
  el.readerPrev = qs("reader-prev");
  el.readerNext = qs("reader-next");
  el.sheetBackdrop = qs("sheet-backdrop");
  el.settingsSheet = qs("settings-sheet");
  el.fontMinus = qs("font-minus");
  el.fontPlus = qs("font-plus");
  el.fontValue = qs("font-value");
  el.settingsClose = qs("settings-close");
  el.themeControls = qs("theme-controls");
}

function applyTheme(choice) {
  state.theme = choice;
  localStorage.setItem("manual.theme", choice);
  if (choice === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", choice);
  }
  if (el.themeControls) {
    el.themeControls.querySelectorAll("button").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.themeChoice === choice);
    });
  }
}

function wireEvents() {
  el.searchInput.addEventListener("input", onSearchInput);
  el.clearBtn.addEventListener("click", () => {
    el.searchInput.value = "";
    onSearchInput();
    el.searchInput.focus();
  });
  el.tocBtn.addEventListener("click", renderTOC);
  el.settingsBtn.addEventListener("click", () => toggleSheet(true));
  el.settingsClose.addEventListener("click", () => toggleSheet(false));
  el.sheetBackdrop.addEventListener("click", () => toggleSheet(false));
  el.fontMinus.addEventListener("click", () => adjustFont(-0.1));
  el.fontPlus.addEventListener("click", () => adjustFont(0.1));
  el.readerClose.addEventListener("click", closeReader);
  el.readerPrev.addEventListener("click", () => stepChunk(-1));
  el.readerNext.addEventListener("click", () => stepChunk(1));
  el.readerViewImage.addEventListener("click", () => {
    const container = el.readerBody.querySelector("#reader-page-image");
    if (!container) return;
    const page = parseInt(el.readerViewImage.dataset.page, 10);
    if (!page) return;
    // Toggle: a second tap on an already-rendered page collapses it again.
    if (el.readerViewImage.dataset.rendered === String(page)) {
      container.innerHTML = "";
      el.readerViewImage.dataset.rendered = "";
      return;
    }
    el.readerViewImage.dataset.rendered = String(page);
    // The preview build doesn't ship the source PDF or the pdf.js renderer
    // (both are far too large for it) — say so plainly instead of trying
    // and silently going nowhere.
    if (state.isPreview) {
      container.innerHTML =
        `<div class="page-image-status">This live preview doesn't include the source PDF, so it can't render the page image here. ` +
        `The installable offline app ships with the PDF cached, so "View page" shows the real page there.</div>`;
      container.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }
    renderPageImage(page, container).then(() => {
      container.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  });
  el.themeControls.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => applyTheme(btn.dataset.themeChoice));
  });
}

function updateOnlineStatus() {
  const online = navigator.onLine;
  el.statusDot.classList.toggle("offline", !online);
  el.statusText.textContent = online
    ? "Online — but works fully offline"
    : "Offline — everything still works";
}

function onSearchInput() {
  const q = el.searchInput.value.trim();
  el.clearBtn.classList.toggle("show", q.length > 0);
  if (!q) {
    renderEmptyState();
    return;
  }
  if (!state.engine) return;
  const { results, queryTokens } = state.engine.search(q, 30);
  state.lastResults = results;
  state.lastQueryTokens = queryTokens;
  renderResults(q, results);
}

function saveRecent(q) {
  const r = state.recent.filter((x) => x !== q);
  r.unshift(q);
  state.recent = r.slice(0, 8);
  localStorage.setItem("manual.recent", JSON.stringify(state.recent));
}

function renderEmptyState() {
  const recentHtml = state.recent.length
    ? `<div class="result-count">Recent searches</div>` +
      state.recent
        .map(
          (q) =>
            `<button class="toc-item" data-recent="${escapeHtml(q)}">🔎 ${escapeHtml(q)}</button>`
        )
        .join("")
    : "";
  el.results.innerHTML = `
    <div class="empty-state">
      <div class="big-icon">📘</div>
      <div><strong>Search the manual.</strong></div>
      <div>Every result is an exact excerpt from the source document — nothing is generated or guessed.</div>
    </div>
    ${recentHtml}
  `;
  el.results.querySelectorAll("[data-recent]").forEach((btn) => {
    btn.addEventListener("click", () => {
      el.searchInput.value = btn.dataset.recent;
      onSearchInput();
    });
  });
}

function renderResults(query, results) {
  if (results.length === 0) {
    el.results.innerHTML = `<div class="empty-state"><div class="big-icon">🤷</div><div><strong>No matches in the manual.</strong></div><div>Try different or fewer words.</div></div>`;
    return;
  }
  el.results.innerHTML =
    `<div class="result-count">${results.length} match${results.length === 1 ? "" : "es"} in the manual</div>` +
    results
      .map((r, i) => {
        const heading = r.chunk.heading || state.engine.meta.sourceFile || "Manual";
        const imageBadge = state.pagesWithImages.has(r.chunk.page) ? " 📷" : "";
        return `
        <button class="result-card" data-idx="${i}">
          <div class="result-meta">
            <span class="result-heading">${escapeHtml(heading)}</span>
            <span class="result-page">p. ${r.chunk.page}${imageBadge}</span>
          </div>
          <div class="result-snippet">${highlight(r.snippet, state.lastQueryTokens)}</div>
        </button>`;
      })
      .join("");
  el.results.querySelectorAll(".result-card").forEach((btn) => {
    btn.addEventListener("click", () => {
      saveRecent(query);
      openReader(parseInt(btn.dataset.idx, 10));
    });
  });
}

/* ---------- Table of contents ----------
   Two-step drill-down rather than one long nested list: first a picker of
   just the top-level chapters as buttons, then — only once one is tapped —
   that chapter's own sections/sub-sections. Keeps the initial view short
   and scannable on a 700+ page manual instead of dumping the whole outline
   at once. */
function renderTOC() {
  const outline = state.engine?.meta?.outline || [];
  if (!outline.length) {
    el.results.innerHTML = `<div class="empty-state"><div class="big-icon">📑</div><div><strong>No table of contents found</strong></div><div>This manual has no embedded bookmarks. Use search instead.</div></div>`;
    el.searchInput.value = "";
    return;
  }
  el.searchInput.value = "";
  el.clearBtn.classList.remove("show");
  renderTOCChapterList(outline);
}

function renderTOCChapterList(outline) {
  el.results.innerHTML =
    `<div class="result-count">Table of contents — select a chapter</div>` +
    `<div class="toc-chapter-list">` +
    outline
      .map(
        (chapter, i) => `
        <button class="toc-chapter-btn" data-chapter-idx="${i}">
          <span class="toc-chapter-title">${escapeHtml(chapter.title || "Untitled")}</span>
          ${chapter.page ? `<span class="toc-page">p. ${chapter.page}</span>` : ""}
          <span class="toc-chevron">›</span>
        </button>`
      )
      .join("") +
    `</div>`;
  el.results.querySelectorAll(".toc-chapter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.dataset.chapterIdx, 10);
      renderTOCChapterDetail(outline, idx);
    });
  });
}

function renderTOCChapterDetail(outline, chapterIdx) {
  const chapter = outline[chapterIdx];
  if (!chapter) return;

  function renderItems(items, depth) {
    return items
      .map((item) => {
        const children = item.children && item.children.length ? renderItems(item.children, depth + 1) : "";
        const pageAttr = item.page ? `data-page="${item.page}"` : "";
        return `
          <button class="toc-item" style="padding-left:${4 + depth * 14}px" ${pageAttr}>
            ${escapeHtml(item.title || "Untitled")}
            ${item.page ? `<span class="toc-page">p. ${item.page}</span>` : ""}
          </button>
          <div class="toc-children">${children}</div>`;
      })
      .join("");
  }

  const hasChildren = chapter.children && chapter.children.length;
  el.results.innerHTML =
    `<button class="toc-back-btn" id="toc-back-btn">‹ All chapters</button>` +
    `<div class="result-count">${escapeHtml(chapter.title || "Untitled")}</div>` +
    (chapter.page
      ? `<button class="toc-chapter-open" data-page="${chapter.page}">
           Jump to start of chapter <span class="toc-page">p. ${chapter.page}</span>
         </button>`
      : "") +
    (hasChildren
      ? `<div class="toc-list">${renderItems(chapter.children, 0)}</div>`
      : `<div class="empty-state"><div>No sub-sections listed for this chapter.</div></div>`);

  document.getElementById("toc-back-btn").addEventListener("click", () => renderTOCChapterList(outline));
  el.results.querySelectorAll(".toc-item[data-page], .toc-chapter-open[data-page]").forEach((btn) => {
    btn.addEventListener("click", () => openPageInReader(parseInt(btn.dataset.page, 10)));
  });
}

function openPageInReader(page) {
  let idx = state.engine.chunks.findIndex((c) => c.page === page);
  if (idx === -1) {
    // Some pages (chapter dividers/cover pages, blank pages) have no
    // extractable text, so no chunk was indexed for them — fall back to
    // the nearest following indexed page rather than the link doing
    // nothing (chunks are in page order from the build, so this is the
    // closest page at or after the one requested).
    idx = state.engine.chunks.findIndex((c) => c.page >= page);
  }
  if (idx === -1) return;
  state.lastResults = [{ chunk: state.engine.chunks[idx], snippet: "" }];
  state.lastQueryTokens = [];
  openReader(0);
}

/* ---------- Reader overlay ---------- */
function openReader(idx) {
  state.readerIdx = idx;
  const r = state.lastResults[idx];
  if (!r) return;
  const chunk = r.chunk;
  const hasImage = state.pagesWithImages.has(chunk.page);
  el.readerTitle.textContent = `Page ${chunk.page}` + (hasImage ? " 📷" : "");
  el.readerBody.innerHTML =
    `<div class="reader-heading">${escapeHtml(chunk.heading || "")}</div>` +
    highlight(chunk.text, state.lastQueryTokens) +
    `<div id="reader-page-image"></div>`;

  // Always show the button — even in the preview build, which lacks the
  // cached source PDF + vendored pdf.js — so a tap explains why it can't
  // render here instead of silently doing nothing (see the click handler).
  el.readerViewImage.style.display = "";
  el.readerViewImage.textContent = hasImage ? "📷 View page" : "View page";
  el.readerViewImage.dataset.page = String(chunk.page);
  el.readerViewImage.dataset.rendered = "";

  el.reader.classList.add("open");
  el.readerBody.scrollTop = 0;
  el.readerPrev.disabled = idx <= 0;
  el.readerNext.disabled = idx >= state.lastResults.length - 1;

  // Keep the left-hand list's selection state in sync (visible on wide
  // screens where both panes are on screen at once; harmless on mobile).
  el.results.querySelectorAll(".result-card.active").forEach((b) => b.classList.remove("active"));
  const activeCard = el.results.querySelector(`.result-card[data-idx="${idx}"]`);
  if (activeCard) activeCard.classList.add("active");
}

function stepChunk(delta) {
  const next = state.readerIdx + delta;
  if (next < 0 || next >= state.lastResults.length) return;
  openReader(next);
}

function closeReader() {
  el.reader.classList.remove("open");
  resetReader();
}

// The empty state shown in the reading pane before anything is selected —
// most relevant on wide screens, where the pane is always visible rather
// than a mobile overlay that's off-screen until opened.
function resetReader() {
  state.readerIdx = -1;
  el.readerTitle.textContent = "";
  el.readerBody.innerHTML = `<div class="empty-state"><div class="big-icon">📖</div><div><strong>Select a result to read it here.</strong></div><div>The full excerpt, with citations, opens on this side.</div></div>`;
  el.readerViewImage.style.display = "none";
  el.readerPrev.disabled = true;
  el.readerNext.disabled = true;
  el.results.querySelectorAll(".result-card.active").forEach((b) => b.classList.remove("active"));
}

/* ---------- In-app page image viewer (shows diagrams/tables exactly as
   printed, rendered from the cached source PDF via pdf.js — no network,
   no relying on the phone's own PDF handling, which mobile browsers are
   inconsistent about). Loaded lazily so pages that never need it don't
   pay for it. ---------- */
let pdfjsLoadPromise = null;
function loadPdfJs() {
  if (pdfjsLoadPromise) return pdfjsLoadPromise;
  pdfjsLoadPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "vendor/pdf.min.js";
    script.onload = () => {
      const lib = window.pdfjsLib;
      lib.GlobalWorkerOptions.workerSrc = "vendor/pdf.worker.min.js";
      resolve(lib);
    };
    script.onerror = () => reject(new Error("Could not load the PDF renderer"));
    document.head.appendChild(script);
  });
  return pdfjsLoadPromise;
}

let pdfDocPromise = null;
function loadPdfDocument() {
  if (pdfDocPromise) return pdfDocPromise;
  pdfDocPromise = loadPdfJs().then((lib) => lib.getDocument("data/manual.pdf").promise);
  return pdfDocPromise;
}

async function renderPageImage(pageNum, container) {
  container.innerHTML = `<div class="page-image-status"><div class="spinner"></div><div>Rendering page ${pageNum}…</div></div>`;
  try {
    const pdf = await loadPdfDocument();
    const page = await pdf.getPage(pageNum);
    const unscaled = page.getViewport({ scale: 1 });
    const displayWidth = Math.min(container.clientWidth || 360, 720);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const scale = (displayWidth * dpr) / unscaled.width;
    const viewport = page.getViewport({ scale });

    const canvas = document.createElement("canvas");
    canvas.width = Math.ceil(viewport.width);
    canvas.height = Math.ceil(viewport.height);
    canvas.style.width = `${Math.ceil(viewport.width / dpr)}px`;
    canvas.style.height = `${Math.ceil(viewport.height / dpr)}px`;
    canvas.className = "page-image-canvas";

    await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
    container.innerHTML = "";
    container.appendChild(canvas);
  } catch (err) {
    container.innerHTML = `<div class="page-image-status">Couldn't render this page (${escapeHtml(
      String(err.message || err)
    )}).</div>`;
  }
}

/* ---------- Settings sheet ---------- */
function toggleSheet(open) {
  el.sheetBackdrop.classList.toggle("open", open);
  el.settingsSheet.classList.toggle("open", open);
}

function adjustFont(delta) {
  state.fontScale = Math.min(1.6, Math.max(0.8, +(state.fontScale + delta).toFixed(2)));
  document.documentElement.style.setProperty("--font-scale", state.fontScale);
  localStorage.setItem("manual.fontScale", String(state.fontScale));
  el.fontValue.textContent = `${Math.round(state.fontScale * 100)}%`;
}

/* ---------- Service worker + install prompt ---------- */
function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(() => {
      /* offline-first app still works without SW registration succeeding */
    });
  }
}

let deferredInstallEvent = null;
function wireInstallPrompt() {
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferredInstallEvent = e;
    showInstallToast();
  });
}

function showInstallToast() {
  if (document.getElementById("install-toast")) return;
  const toast = document.createElement("div");
  toast.id = "install-toast";
  toast.className = "install-toast";
  toast.innerHTML = `<span>Install this manual for one-tap offline access.</span>
    <button id="install-yes">Install</button>
    <button class="dismiss" id="install-no">Not now</button>`;
  document.body.appendChild(toast);
  document.getElementById("install-yes").addEventListener("click", async () => {
    toast.remove();
    if (deferredInstallEvent) {
      deferredInstallEvent.prompt();
      await deferredInstallEvent.userChoice;
      deferredInstallEvent = null;
    }
  });
  document.getElementById("install-no").addEventListener("click", () => toast.remove());
}

document.addEventListener("DOMContentLoaded", boot);
