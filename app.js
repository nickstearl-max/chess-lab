// Chess Lab for iPhone: screenshot/photo -> board vision (Web Worker) -> Stockfish (WASM worker).

import { Engine, formatEval, shortEval, winFraction } from "./engine.js";
import { PIECE_SVG } from "./pieces.js";
import { describeLine, isLegalFen, squareToGrid } from "./rules.js";

const $ = id => document.getElementById(id);
const MAX_SIDE = 2000;               // screenshots are smaller; photos get scaled down to this
const SQ = 50;                       // board SVG square size (viewBox 400)

// ------------------------------------------------------------------ settings (per device)
const store = {
  get(key, fallback) {
    try {
      const v = localStorage.getItem("chesslab." + key);
      return v === null ? fallback : JSON.parse(v);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try { localStorage.setItem("chesslab." + key, JSON.stringify(value)); } catch { /* private mode */ }
  },
};

const settings = {
  orientation: store.get("orientation", "auto"),
  turn: store.get("turn", "auto"),
  depth: store.get("depth", 16),
  skill: store.get("skill", 20),
  setId: store.get("setId", "default"),
};

// ------------------------------------------------------------------ calibrated sets (IndexedDB)
const idb = {
  open() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open("chesslab", 1);
      req.onupgradeneeded = () => req.result.createObjectStore("sets", { keyPath: "id" });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  },
  async run(mode, fn) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction("sets", mode);
      const req = fn(tx.objectStore("sets"));
      tx.oncomplete = () => resolve(req && req.result);
      tx.onerror = () => reject(tx.error);
    });
  },
  all() { return this.run("readonly", s => s.getAll()).catch(() => []); },
  put(rec) { return this.run("readwrite", s => s.put(rec)); },
  delete(id) { return this.run("readwrite", s => s.delete(id)); },
};

// ------------------------------------------------------------------ workers
const vision = new Worker("./vision-worker.js", { type: "module" });
const pending = new Map();
let nextId = 1;
vision.onmessage = e => {
  const { id, result, error } = e.data;
  const p = pending.get(id);
  if (!p) return;
  pending.delete(id);
  error ? p.reject(new Error(error)) : p.resolve(result);
};
vision.onerror = e => toast("Vision worker failed: " + (e.message || "unknown error"));
function call(type, payload = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    vision.postMessage({ id, type, ...payload }, transfer);
  });
}

const engine = new Engine("./vendor/stockfish-19-lite-single.js");

// ------------------------------------------------------------------ state
let sets = [];                // {id, label, symbols, file?, user?}
const loadedSets = new Set();
let image = null;             // {canvas, w, h}  downscaled source, for corner dragging
let reading = null;           // last vision result
let analysis = null;          // {fen, first, san, cp, mate, depth, final}
let sticky = null;            // orientation memory for ambiguous boards
let view = "board";
let corners = null;           // manual corners, kept until a new image arrives

// ------------------------------------------------------------------ UI helpers
let toastTimer = 0;
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
}

function busy(text) {
  $("busy").hidden = !text;
  if (text) $("busyText").textContent = text;
}

function setEngineChip(state, text) {
  $("engineChip").className = "chip " + state;
  $("engineText").textContent = text;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

// ------------------------------------------------------------------ board drawing
const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

function placementMap(placement) {
  const map = {};
  (placement || "").split("/").forEach((row, i) => {
    let file = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) file += Number(ch);
      else { map["abcdefgh"[file] + (8 - i)] = ch; file++; }
    }
  });
  return map;
}

// arrow points in a board of `sq`-sized squares; knights get an L, like the desktop overlay
function arrowGeometry(move, wab, sq) {
  const centre = name => { const [r, c] = squareToGrid(name, wab); return [(c + 0.5) * sq, (r + 0.5) * sq]; };
  const [x1, y1] = centre(move.from), [x2, y2] = centre(move.to);
  const pts = [[x1, y1]];
  if (move.piece && move.piece.toLowerCase() === "n") pts.push(Math.abs(x2 - x1) > Math.abs(y2 - y1) ? [x2, y1] : [x1, y2]);
  const head = sq * 0.42, [px, py] = pts[pts.length - 1];
  const len = Math.hypot(x2 - px, y2 - py) || 1, ux = (x2 - px) / len, uy = (y2 - py) / len;
  const bx = x2 - ux * head, by = y2 - uy * head, w = sq * 0.3;
  pts.push([bx, by]);
  return { pts, tip: [x2, y2], left: [bx - uy * w, by + ux * w], right: [bx + uy * w, by - ux * w], width: sq * 0.2 };
}

function drawBoard() {
  const svg = $("board");
  svg.innerHTML = "";
  const wab = reading?.orientation ? reading.orientation.whiteAtBottom : true;
  const fen = analysis?.fen || reading?.fen || "";
  const placement = fen ? fen.split(" ")[0] : reading?.placement || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
  const map = placementMap(placement);
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      svgEl("rect", { x: c * SQ, y: r * SQ, width: SQ, height: SQ, fill: (r + c) % 2 ? "var(--dark-sq)" : "var(--light-sq)" }, svg);
    }
  for (let i = 0; i < 8; i++) {                       // coordinates
    const file = "abcdefgh"[wab ? i : 7 - i], rank = wab ? 8 - i : i + 1;
    svgEl("text", { x: i * SQ + SQ - 3, y: 400 - 3, "text-anchor": "end", "font-size": 9, "font-weight": 700,
      fill: i % 2 ? "var(--light-sq)" : "var(--dark-sq)" }, svg).textContent = file;
    svgEl("text", { x: 3, y: i * SQ + 11, "font-size": 9, "font-weight": 700,
      fill: i % 2 ? "var(--light-sq)" : "var(--dark-sq)" }, svg).textContent = rank;
  }
  const move = analysis?.first;
  if (move) {
    for (const sq of [move.from, move.to]) {
      const [r, c] = squareToGrid(sq, wab);
      svgEl("rect", { x: c * SQ + 2, y: r * SQ + 2, width: SQ - 4, height: SQ - 4, rx: 4, fill: "var(--accent)",
        "fill-opacity": 0.22, stroke: "var(--accent)", "stroke-width": 3 }, svg);
    }
  }
  for (const [sq, sym] of Object.entries(map)) {
    const [r, c] = squareToGrid(sq, wab);
    const g = svgEl("g", { transform: `translate(${c * SQ} ${r * SQ}) scale(${SQ / 45})` }, svg);
    g.innerHTML = PIECE_SVG[sym] || "";
  }
  for (const sq of reading?.ok === false ? reading.uncertain || [] : []) {
    const [r, c] = squareToGrid(sq, wab);
    svgEl("rect", { x: c * SQ, y: r * SQ, width: SQ, height: SQ, fill: "var(--warn)", "fill-opacity": 0.35 }, svg);
    svgEl("text", { x: c * SQ + SQ / 2, y: r * SQ + SQ / 2 + 11, "text-anchor": "middle", "font-size": 32,
      "font-weight": 800, fill: "var(--warn)", stroke: "#000", "stroke-width": 1 }, svg).textContent = "?";
  }
  if (move) {
    const a = arrowGeometry(move, wab, SQ);
    const g = svgEl("g", { opacity: 0.92 }, svg);
    svgEl("polyline", { points: a.pts.map(p => p.join(",")).join(" "), fill: "none", stroke: "var(--accent)",
      "stroke-width": a.width, "stroke-linecap": "round", "stroke-linejoin": "round" }, g);
    svgEl("polygon", { points: [a.tip, a.left, a.right].map(p => p.join(",")).join(" "), fill: "var(--accent)" }, g);
  }
}

function drawShot() {
  const cv = $("shot"), ctx = cv.getContext("2d");
  ctx.fillStyle = "#0e151d";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!reading?.warped) return;
  ctx.putImageData(new ImageData(new Uint8ClampedArray(reading.warped), reading.warpedSize, reading.warpedSize), 0, 0);
  const move = analysis?.first, wab = reading.orientation ? reading.orientation.whiteAtBottom : true, sq = cv.width / 8;
  for (const name of reading.ok === false ? reading.uncertain || [] : []) {
    const [r, c] = squareToGrid(name, wab);
    ctx.fillStyle = "rgba(255,179,71,0.45)";
    ctx.fillRect(c * sq, r * sq, sq, sq);
  }
  if (!move) return;
  const a = arrowGeometry(move, wab, sq);
  ctx.strokeStyle = ctx.fillStyle = "#39f58c";
  ctx.globalAlpha = 0.9;
  ctx.lineWidth = a.width;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.beginPath();
  a.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(...a.tip); ctx.lineTo(...a.left); ctx.lineTo(...a.right);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
}

function render() {
  $("empty").hidden = Boolean(reading || analysis);
  $("board").hidden = view !== "board";
  $("shot").hidden = view !== "shot";
  $("viewBoard").classList.toggle("on", view === "board");
  $("viewShot").classList.toggle("on", view === "shot");
  $("viewShot").disabled = !reading?.warped;
  drawBoard();
  drawShot();

  // result card
  const card = $("resultCard");
  const fen = analysis?.fen || reading?.fen || "";
  const turn = fen ? (fen.split(" ")[1] === "w" ? "White to move" : "Black to move") : "";
  if (analysis?.first) {
    card.hidden = false;
    $("san").className = "san";
    $("san").textContent = analysis.first.san;
    $("evalChip").textContent = shortEval(analysis.cp, analysis.mate);
    $("moveDesc").textContent = analysis.first.desc;
    $("line").textContent = analysis.san.slice(0, 8).join("  ");
    $("meta").textContent = [turn, `depth ${analysis.depth}${analysis.final ? "" : "…"}`, formatEval(analysis.cp, analysis.mate)]
      .filter(Boolean).join(" · ");
  } else if (reading?.gameOver) {
    card.hidden = false;
    $("san").className = "san status";
    $("san").textContent = "Game over";
    $("evalChip").textContent = "";
    $("moveDesc").textContent = reading.gameOver;
    $("line").textContent = "";
    $("meta").textContent = "";
  } else {
    card.hidden = true;
  }
  $("evalWhite").style.width = `${Math.round(winFraction(analysis?.cp ?? null, analysis?.mate ?? null) * 100)}%`;

  // problems
  const banner = $("banner");
  if (reading && !reading.ok) {
    banner.hidden = false;
    banner.className = "banner";
    $("bannerTitle").textContent = "Not sure, so no move";
    $("bannerList").innerHTML = reading.errors.map(e => `<li>${escapeHtml(e)}</li>`).join("");
    $("bannerAdjust").hidden = !image;
  } else if (reading?.warnings?.length) {
    banner.hidden = false;
    banner.className = "banner";
    $("bannerTitle").textContent = "Heads up";
    $("bannerList").innerHTML = reading.warnings.map(e => `<li>${escapeHtml(e)}</li>`).join("");
    $("bannerAdjust").hidden = true;
  } else {
    banner.hidden = true;
  }
  if (fen && document.activeElement !== $("fen")) $("fen").value = fen;
}

// ------------------------------------------------------------------ piece sets
async function refreshSets() {
  let builtIn = [];
  try {
    builtIn = await (await fetch("templates/index.json")).json();
  } catch {
    toast("Couldn't load the piece sets.");
  }
  const user = (await idb.all()).map(s => ({ id: s.id, label: s.label, symbols: s.symbols, user: true }));
  sets = [...builtIn, ...user];
  if (!sets.some(s => s.id === settings.setId)) settings.setId = "default";
  const sel = $("pieceSet");
  sel.innerHTML = sets.map(s => `<option value="${escapeHtml(s.id)}">${escapeHtml(s.label)}</option>`).join("");
  sel.value = settings.setId;
  $("deleteSet").hidden = !sets.find(s => s.id === settings.setId)?.user;
}

async function ensureSet(id) {
  if (loadedSets.has(id)) return;
  const set = sets.find(s => s.id === id);
  if (!set) throw new Error("Unknown piece set");
  let bytes;
  if (set.user) {
    const rec = (await idb.all()).find(s => s.id === id);
    bytes = rec.bytes.slice(0);
  } else {
    bytes = await (await fetch(set.file)).arrayBuffer();
  }
  await call("load-set", { setId: id, label: set.label, symbols: set.symbols, bytes }, [bytes]);
  loadedSets.add(id);
}

// ------------------------------------------------------------------ images
async function decode(blob) {
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.decoding = "async";
    img.src = url;
    await img.decode();
    const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    const w = Math.max(1, Math.round(img.naturalWidth * scale)), h = Math.max(1, Math.round(img.naturalHeight * scale));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, w, h);
    return { canvas, w, h };
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function loadImage(blob) {
  if (!blob) return;
  busy("Opening image…");
  try {
    image = await decode(blob);
    corners = null;
    const data = image.canvas.getContext("2d").getImageData(0, 0, image.w, image.h).data;
    await call("image", { w: image.w, h: image.h, data: data.buffer }, [data.buffer]);
    await readBoard();
  } catch (err) {
    busy(null);
    toast("Couldn't open that image: " + err.message);
  }
}

async function readBoard() {
  if (!image) return;
  busy("Reading board…");
  analysis = null;
  try {
    await ensureSet(settings.setId);
    reading = await call("read", {
      setId: settings.setId, corners,
      options: { orientation: settings.orientation, turn: settings.turn, sticky },
    });
  } catch (err) {
    busy(null);
    toast(err.message);
    return;
  }
  if (reading.orientation?.method === "pieces" && reading.orientation.confidence >= 0.5) {
    sticky = reading.orientation.whiteAtBottom;
  }
  busy(null);
  render();
  if (reading.ok && !reading.gameOver) analyse(reading.fen);
}

async function analyse(fen) {
  busy("Stockfish thinking…");
  const base = { fen, first: null, san: [], cp: null, mate: null, depth: 0, final: false };
  const show = (info, final) => {
    const line = describeLine(fen, info.bestmove ? [info.bestmove, ...info.pv.slice(1)] : info.pv);
    if (!line.first) return;
    analysis = { ...base, ...line, cp: info.cp, mate: info.mate, depth: info.depth, final };
    render();
  };
  try {
    const result = await engine.analyse(fen, {
      depth: settings.depth, skill: settings.skill, onInfo: info => { if (info.depth >= 4) show(info, false); },
    });
    if (result === null) return;                      // superseded by a newer analysis
    busy(null);
    if (!result.bestmove) {
      analysis = null;
      toast("No legal move in this position.");
      render();
      return;
    }
    show(result, true);
  } catch (err) {
    busy(null);
    setEngineChip("error", "Engine error");
    toast("Stockfish error: " + err.message);
  }
}

// ------------------------------------------------------------------ corner dragging
const adjust = { quad: null, drag: -1, scale: 1 };

function openAdjust() {
  if (!image) { toast("Load a screenshot first."); return; }
  const det = reading?.detection;
  adjust.quad = (corners || det?.corners || [[image.w * 0.1, image.h * 0.1], [image.w * 0.9, image.h * 0.1],
    [image.w * 0.9, image.h * 0.9], [image.w * 0.1, image.h * 0.9]]).map(p => [...p]);
  $("adjust").hidden = false;
  const cv = $("adjustCanvas");
  cv.width = image.w;
  cv.height = image.h;
  drawAdjust();
}

function drawAdjust() {
  const cv = $("adjustCanvas"), ctx = cv.getContext("2d");
  ctx.drawImage(image.canvas, 0, 0);
  const q = adjust.quad, r = Math.max(image.w, image.h) / 40;
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.beginPath();
  ctx.rect(0, 0, cv.width, cv.height);
  ctx.moveTo(...q[0]); q.slice(1).forEach(p => ctx.lineTo(...p)); ctx.closePath();
  ctx.fill("evenodd");
  ctx.strokeStyle = "#39f58c";
  ctx.lineWidth = Math.max(2, r / 5);
  ctx.beginPath();
  ctx.moveTo(...q[0]); q.slice(1).forEach(p => ctx.lineTo(...p)); ctx.closePath();
  ctx.stroke();
  for (const [x, y] of q) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(57,245,140,0.35)";
    ctx.fill();
    ctx.stroke();
  }
}

function canvasPoint(e) {
  const cv = $("adjustCanvas"), rect = cv.getBoundingClientRect();
  return [(e.clientX - rect.left) * cv.width / rect.width, (e.clientY - rect.top) * cv.height / rect.height];
}

$("adjustCanvas").addEventListener("pointerdown", e => {
  const [x, y] = canvasPoint(e), cv = $("adjustCanvas");
  const reach = Math.max(image.w, image.h) / 12;
  let best = -1, bestD = Infinity;
  adjust.quad.forEach(([qx, qy], i) => { const d = Math.hypot(qx - x, qy - y); if (d < bestD) { bestD = d; best = i; } });
  if (bestD < reach) {
    adjust.drag = best;
    cv.setPointerCapture(e.pointerId);
  }
});
$("adjustCanvas").addEventListener("pointermove", e => {
  if (adjust.drag < 0) return;
  const [x, y] = canvasPoint(e);
  adjust.quad[adjust.drag] = [Math.max(0, Math.min(image.w, x)), Math.max(0, Math.min(image.h, y))];
  drawAdjust();
});
for (const type of ["pointerup", "pointercancel"]) {
  $("adjustCanvas").addEventListener(type, () => { adjust.drag = -1; });
}
$("adjustCancel").addEventListener("click", () => { $("adjust").hidden = true; });
$("adjustUse").addEventListener("click", () => {
  corners = adjust.quad;
  $("adjust").hidden = true;
  readBoard();
});

// ------------------------------------------------------------------ controls
function syncControls() {
  document.querySelectorAll(".segmented").forEach(seg => {
    const key = seg.dataset.setting;
    seg.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.value === settings[key]));
  });
  $("depth").value = settings.depth;
  $("depthOut").textContent = settings.depth;
  $("skill").value = settings.skill;
  $("skillOut").textContent = settings.skill === 20 ? "20 (full)" : settings.skill;
}

document.querySelectorAll(".segmented").forEach(seg => {
  seg.addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    settings[seg.dataset.setting] = b.dataset.value;
    store.set(seg.dataset.setting, b.dataset.value);
    syncControls();
    readBoard();
  });
});

for (const key of ["depth", "skill"]) {
  $(key).addEventListener("input", () => {
    settings[key] = Number($(key).value);
    store.set(key, settings[key]);
    syncControls();
  });
  $(key).addEventListener("change", () => {
    const fen = analysis?.fen || (reading?.ok ? reading.fen : "");
    if (fen) analyse(fen);
  });
}

$("pieceSet").addEventListener("change", () => {
  settings.setId = $("pieceSet").value;
  store.set("setId", settings.setId);
  $("deleteSet").hidden = !sets.find(s => s.id === settings.setId)?.user;
  readBoard();
});

$("deleteSet").addEventListener("click", async () => {
  const set = sets.find(s => s.id === settings.setId);
  if (!set?.user || !confirm(`Delete the piece set "${set.label}"?`)) return;
  await idb.delete(set.id);
  loadedSets.delete(set.id);
  settings.setId = "default";
  store.set("setId", "default");
  await refreshSets();
  readBoard();
});

$("calibrateBtn").addEventListener("click", async () => {
  const name = $("calName").value.trim() || "My pieces";
  if (!image) { toast("First load a screenshot of a board in the starting position."); return; }
  busy("Learning the pieces…");
  try {
    if (!reading?.detection) await readBoard();
    const res = await call("calibrate", { name, orientation: settings.orientation });
    const id = "user_" + Date.now().toString(36);
    await idb.put({ id, label: name, symbols: res.symbols, bytes: res.bytes });
    settings.setId = id;
    store.set("setId", id);
    await refreshSets();
    $("calName").value = "";
    toast(`Saved piece set "${name}".`);
    await readBoard();
  } catch (err) {
    busy(null);
    toast(err.message);
  }
});

$("fileInput").addEventListener("change", e => {
  const file = e.target.files && e.target.files[0];
  e.target.value = "";
  loadImage(file);
});

$("pasteBtn").addEventListener("click", async () => {
  try {
    const items = await navigator.clipboard.read();
    for (const item of items) {
      const type = item.types.find(t => t.startsWith("image/"));
      if (type) { loadImage(await item.getType(type)); return; }
    }
    toast("No image on the clipboard. Copy a screenshot first.");
  } catch {
    toast("Paste isn't allowed here. Use Screenshot or photo instead.");
  }
});

document.addEventListener("paste", e => {
  const file = [...(e.clipboardData?.files || [])].find(f => f.type.startsWith("image/"));
  if (file) loadImage(file);
});

$("viewBoard").addEventListener("click", () => { view = "board"; render(); });
$("viewShot").addEventListener("click", () => { if (reading?.warped) { view = "shot"; render(); } });
$("adjustBtn").addEventListener("click", openAdjust);
$("bannerAdjust").addEventListener("click", openAdjust);

$("analyseFen").addEventListener("click", () => {
  const fen = $("fen").value.trim();
  const problem = isLegalFen(fen);
  if (problem) { toast(problem); return; }
  const full = fen.split(/\s+/).length >= 4 ? fen : `${fen.split(/\s+/).slice(0, 2).join(" ")} - - 0 1`;
  reading = { ok: true, fen: full, placement: full.split(" ")[0], errors: [], warnings: [], uncertain: [],
              orientation: reading?.orientation || { whiteAtBottom: full.split(" ")[1] !== "b" } };
  view = "board";
  analyse(full);
  render();
});

$("copyFen").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("fen").value.trim());
    toast("FEN copied.");
  } catch {
    $("fen").select();
    toast("Select and copy the FEN above.");
  }
});

// ------------------------------------------------------------------ start
async function start() {
  syncControls();
  render();
  if ("serviceWorker" in navigator && !["localhost", "127.0.0.1"].includes(location.hostname)) {
    navigator.serviceWorker.register("./sw.js").catch(() => { /* http or private mode: no offline copy */ });
  }
  await refreshSets();
  ensureSet(settings.setId).catch(() => {});
  try {
    await engine.init();
    const version = (engine.name.match(/Stockfish\s+(\d+)/) || [])[1];
    setEngineChip("ready", `Stockfish${version ? " " + version : ""} ready`);
  } catch (err) {
    setEngineChip("error", "Stockfish failed");
    toast(err.message);
  }
}

start();
